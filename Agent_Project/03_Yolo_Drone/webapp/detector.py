"""
YOLO 无人机检测器封装模块
提供图片、视频、摄像头的统一检测接口
"""
import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class DroneDetector:
    """无人机检测器核心类"""

    HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection_history.json')

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'drone', 1: 'bird'}
        self.detection_history = []
        self.stats = {
            'total_detections': 0,
            'drone_count': 0,
            'bird_count': 0,
            'images_processed': 0,
            'videos_processed': 0,
            'avg_inference_time': 0,
            'inference_times': [],
        }
        self._load_history()

    def _load_history(self):
        """从文件加载历史记录"""
        if os.path.exists(self.HISTORY_FILE):
            try:
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.detection_history = data.get('history', [])
                saved_stats = data.get('stats', {})
                # 恢复统计（inference_times 列表可能很大，只恢复均值）
                self.stats['total_detections'] = saved_stats.get('total_detections', 0)
                self.stats['drone_count'] = saved_stats.get('drone_count', 0)
                self.stats['bird_count'] = saved_stats.get('bird_count', 0)
                self.stats['images_processed'] = saved_stats.get('images_processed', 0)
                self.stats['videos_processed'] = saved_stats.get('videos_processed', 0)
                self.stats['avg_inference_time'] = saved_stats.get('avg_inference_time', 0)
                print(f"[*] 已加载 {len(self.detection_history)} 条历史记录")
            except Exception as e:
                print(f"[!] 加载历史失败: {e}")

    def _save_history(self):
        """保存历史记录到文件"""
        try:
            data = {
                'history': self.detection_history,
                'stats': {
                    'total_detections': self.stats['total_detections'],
                    'drone_count': self.stats['drone_count'],
                    'bird_count': self.stats['bird_count'],
                    'images_processed': self.stats['images_processed'],
                    'videos_processed': self.stats['videos_processed'],
                    'avg_inference_time': self.stats['avg_inference_time'],
                }
            }
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] 保存历史失败: {e}")

    def detect_image(self, image_path, conf=0.25, save=True, save_dir='static/results'):
        """
        检测单张图片
        返回: 原图路径, 结果图路径, 检测信息列表
        """
        start = time.time()
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            project=save_dir,
            exist_ok=True,
            verbose=False
        )
        inference_time = (time.time() - start) * 1000  # ms

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    'class': self.class_names[cls_id],
                    'confidence': round(conf_val * 100, 1),
                    'bbox': [x1, y1, x2, y2],
                    'area': (x2 - x1) * (y2 - y1),
                })

        # 获取结果图路径
        result_path = None
        if save and results:
            for r in results:
                if r.save_dir:
                    # 结果保存在 save_dir 下，文件名与原图相同
                    orig_name = os.path.basename(image_path)
                    result_path = os.path.join(str(r.save_dir), orig_name)
                    if not os.path.exists(result_path):
                        # 尝试其他可能的路径
                        for f in os.listdir(str(r.save_dir)):
                            if orig_name.rsplit('.', 1)[0] in f:
                                result_path = os.path.join(str(r.save_dir), f)
                                break

        # 如果没找到结果路径，自己绘制
        if result_path is None or not os.path.exists(result_path):
            img = cv2.imread(image_path)
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                color = (0, 255, 0) if d['class'] == 'drone' else (0, 165, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{d['class']} {d['confidence']}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2)
            result_name = f"result_{os.path.basename(image_path)}"
            result_path = os.path.join(save_dir, result_name)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(result_path, img)

        # 更新统计
        self.stats['images_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['drone_count'] += sum(1 for d in detections if d['class'] == 'drone')
        self.stats['bird_count'] += sum(1 for d in detections if d['class'] == 'bird')
        self.stats['inference_times'].append(inference_time)
        self.stats['avg_inference_time'] = round(
            sum(self.stats['inference_times']) / len(self.stats['inference_times']), 1
        )

        # 记录历史
        record = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'image',
            'source': os.path.basename(image_path),
            'detections': len(detections),
            'details': detections,
            'inference_time': round(inference_time, 1),
            'result_path': result_path.replace('\\', '/'),
        }
        self.detection_history.insert(0, record)
        self._save_history()

        return result_path, detections, round(inference_time, 1)

    def detect_video_frame(self, frame, conf=0.25):
        """
        检测视频/摄像头的单帧
        返回: 绘制了检测框的帧, 检测信息列表
        """
        start = time.time()
        results = self.model.predict(
            source=frame,
            conf=conf,
            save=False,
            verbose=False
        )
        inference_time = (time.time() - start) * 1000

        detections = []
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = self.class_names[cls_id]
                detections.append({
                    'class': cls_name,
                    'confidence': round(conf_val * 100, 1),
                    'bbox': [x1, y1, x2, y2],
                })
                # 绘制框
                color = (0, 255, 0) if cls_name == 'drone' else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_val*100:.0f}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 在帧上显示 FPS 信息
        fps = 1000 / max(inference_time, 1)
        info_text = f"FPS: {fps:.0f} | Objects: {len(detections)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 更新统计
        self.stats['total_detections'] += len(detections)
        self.stats['drone_count'] += sum(1 for d in detections if d['class'] == 'drone')
        self.stats['bird_count'] += sum(1 for d in detections if d['class'] == 'bird')

        return annotated_frame, detections, round(inference_time, 1)

    def get_frame_jpeg(self, frame, quality=80):
        """将帧编码为 JPEG 字节"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def get_stats(self):
        """获取检测统计信息"""
        return {
            **self.stats,
            'history_count': len(self.detection_history),
        }

    def get_history(self, limit=50):
        """获取检测历史"""
        return self.detection_history[:limit]

    def clear_history(self):
        """清空检测历史"""
        self.detection_history.clear()
        self.stats = {
            'total_detections': 0,
            'drone_count': 0,
            'bird_count': 0,
            'images_processed': 0,
            'videos_processed': 0,
            'avg_inference_time': 0,
            'inference_times': [],
        }
        self._save_history()
