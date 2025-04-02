import cv2
import time
import torch
import tqdm
import numpy as np
from ultralytics import YOLO
from deep_sort_oh.deepsort_tracker import DeepSort
from SlowFast.slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from SlowFast.slowfast.visualization.predictor import ActionPredictor
from SlowFast.slowfast.visualization.video_visualizer import VideoVisualizer
from SlowFast.slowfast.visualization.demo_loader import VideoManager
from SlowFast.slowfast.utils.parser import load_config, parse_args


class SlowFastDemo:
    def __init__(self, cfg):
        """
        初始化 SlowFast 目标检测和行为识别 demo。
        """
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化 YOLO 检测器
        self.detector = YOLO('./model/yolo11n.pt').to(self.device)
        # 初始化 DeepSORT 追踪器
        self.tracker = DeepSort(max_age=100, nms_max_overlap=0.8, n_init=3)
        # 使用clip模型
        # self.tracker = DeepSort(max_age=30, max_iou_distance=0.5, nms_max_overlap=1, embedder='clip_ViT-B/16', embedder_wts="./model/ViT-B-16.pt")
        # 使用torchreid模型
        # deepsort = DeepSort(max_age=30, max_iou_distance=0.8, nms_max_overlap=0.8, embedder='torchreid')
        
        # 视频可视化工具
        self.video_vis = VideoVisualizer(
            num_classes=cfg.MODEL.NUM_CLASSES,
            class_names_path=cfg.DEMO.LABEL_FILE_PATH,
            top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
            thres=cfg.DEMO.COMMON_CLASS_THRES,
            lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
            common_class_names=(
                cfg.DEMO.COMMON_CLASS_NAMES if len(cfg.DEMO.LABEL_FILE_PATH) != 0 else None
            ),
            colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
            mode=cfg.DEMO.VIS_MODE,
        )
        self.async_vis = AsyncVis(self.video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

        # 初始化 SlowFast 模型
        self.model = ActionPredictor(cfg=cfg, async_vis=self.async_vis)
        
    def process_detections(self, frame):
        """
        目标检测与跟踪。
        """
        results = self.detector(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        human_dets = [
            ([x1, y1, x2 - x1, y2 - y1], conf)
            for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes)
            if conf > 0.25 and cls == 0
        ]
        
        return self.tracker.update_tracks(human_dets, frame=frame) if human_dets else []

    def run_demo(self, frame_provider):
        """
        处理视频帧并进行行为识别。
        """
        frame_provider.start()
        num_task = 0

        for able_to_read, task in frame_provider:
            if not able_to_read:
                break
            if task is None:
                time.sleep(0.02)
                continue
            num_task += 1
            
            self.model.put(task)
            try:
                task = self.model.get()
                num_task -= 1
                yield task
            except IndexError:
                continue

        while num_task != 0:
            try:
                task = self.model.get()
                num_task -= 1
                yield task
            except IndexError:
                continue
    
    def demo(self):
        """
        运行整个 demo 进行视频分析。
        """
        frame_provider = VideoManager(self.cfg)
        frame_count = 0
        
        for task in tqdm.tqdm(self.run_demo(frame_provider)):
            if task is None:
                time.sleep(0.02)
                continue

            for frame in task.frames[task.num_buffer_frames:]:
                print(frame_count)
                frame_count += 1
                if frame is None:
                    break
                
                tracks = self.process_detections(frame)
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    color1 = (0, 255, 0)  # 追踪框颜色
                    color2 = (255, 0, 0)  # ID颜色
                    fontsize = 0.4  # 1.2
                    fontline = 1  # 3
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                    cv2.putText(frame, f"ID:{track.track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, fontsize, color2, fontline)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color1, fontline)
            
            frame_provider.display(task)
        
        frame_provider.join()
        frame_provider.clean()


def main():
    args = parse_args()
    cfg = load_config(args)
    
    if cfg.DEMO.ENABLE:
        demo = SlowFastDemo(cfg)
        demo.demo()


if __name__ == "__main__":
    main()
