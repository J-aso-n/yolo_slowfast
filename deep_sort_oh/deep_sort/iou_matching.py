# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment

import time

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]  # 目标框 bbox 的左上角和右下角坐标
    candidates_tl = candidates[:, :2]  # 所有候选框 candidates 的左上角坐标
    candidates_br = candidates[:, :2] + candidates[:, 2:]  # 所有候选框 candidates 的右下角坐标

    # 计算交集
    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)  # 交集面积
    area_bbox = bbox[2:].prod()  # 检测框面积
    area_candidates = candidates[:, 2:].prod(axis=1)  # 候选框面积
    return area_intersection / (area_bbox + area_candidates - area_intersection)

# 拓展iou算法
def iou_ext_sep(bb_det, bb_trk, ext_w, ext_h):
    """
    Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,w,h]
    with separate extension coefficient
    """
    # 将[x1,y1,w,h]形状的检测框变为[x1,y1,x2,y2]
    bb_det = [bb_det[0], bb_det[1], bb_det[0]+ bb_det[2], bb_det[1]+bb_det[3]]  
    bb_trk = [bb_trk[0], bb_trk[1], bb_trk[0]+ bb_trk[2], bb_trk[1]+bb_trk[3]]

    trk_w = bb_trk[2] - bb_trk[0]
    trk_h = bb_trk[3] - bb_trk[1]
    xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext_w/2)
    xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext_w/2)
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext_h/2)
    yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext_h/2)
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


# sort_oh算法
# track是Track类
def new_iou(track, tracks, candidates):
    bbox = track.to_ltwh()
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]  # 目标框 bbox 的左上角和右下角坐标
    candidates_tl = candidates[:, :2]  # 所有候选框 candidates 的左上角坐标
    candidates_br = candidates[:, :2] + candidates[:, 2:]  # 所有候选框 candidates 的右下角坐标

    # 计算交集
    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)  # track和detection交集面积
    area_bbox = bbox[2:].prod()  # track检测框面积
    area_candidates = candidates[:, 2:].prod(axis=1)  # candidates候选框面积
    # 计算Ci遮挡置信度
    if track.time_since_update > 1 :
        alpha = 1  # 超参数
        C_i = np.minimum(1, alpha * (track.age / track.time_since_update) * (area_bbox / np.mean(area_candidates)))
        # print("C_i:", C_i)
        # 计算交集CPi
        tracks_tl = tracks[:, :2]  # 所有tracks 的左上角坐标
        tracks_br = tracks[:, :2] + tracks[:, 2:]  # 所有tracks 的右下角坐标
        tl1 = np.c_[
            np.maximum(bbox_tl[0], tracks_tl[:, 0])[:, np.newaxis],
            np.maximum(bbox_tl[1], tracks_tl[:, 1])[:, np.newaxis],
        ]
        br1 = np.c_[
            np.minimum(bbox_br[0], tracks_br[:, 0])[:, np.newaxis],
            np.minimum(bbox_br[1], tracks_br[:, 1])[:, np.newaxis],
        ]
        wh1 = np.maximum(0.0, br1 - tl1)

        intersection = wh1.prod(axis=1)  # track和tracks交集面积
        ratios = intersection / area_bbox
        ratios = ratios[ratios < 1]
        if ratios.size == 0:
            return [1]
        CP_i = np.max(ratios)
        # print("CP_i:", CP_i)
        if C_i > 0.7 and CP_i > 0.8:  # 认定是遮挡目标
            ext_w = np.minimum(1.2, (track.time_since_update + 1) * 0.3)
            ext_h = np.minimum(0.5, (track.time_since_update + 1) * 0.1)
            iou_ext = []
            for candidate in candidates:
                iou_ext_mid = iou_ext_sep(candidate, bbox, ext_w, ext_h)
                iou_ext.append(iou_ext_mid)
            iou_ext = np.array(iou_ext) 
            # print("iou:", [iou_ for iou_ in iou_ext if iou_ > 0 ])
            return iou_ext
    
    return area_intersection / (area_bbox + area_candidates - area_intersection)  # 无遮挡情况


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    tracks_ = np.asarray([tracks[i].to_ltwh() for i in np.arange(len(tracks))])
    # candidates = np.asarray([detections[i].ltwh for i in detection_indices])
    candidates = np.asarray([detections[i].ltwh for i in np.arange(len(detections))])

    for row, track_idx in enumerate(track_indices):
        # bbox = tracks[track_idx].to_ltwh()
        # cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
        iou_mid = new_iou(tracks[track_idx], tracks_, candidates)
        cost_matrix[row, :] = np.asarray([1.0 - iou_mid[i] for i in detection_indices])
    return cost_matrix
