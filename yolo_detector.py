import numpy as np
import time
import math
from typing import List, Dict
from ultralytics import YOLO
from camera_calibration import CameraCalibration
from object_3d_estimator import Object3DEstimator
from detection_3d import Detection3D

class EnhancedYOLODetector:
    """Enhanced YOLO with automotive-specific optimizations"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.object_3d_estimator = Object3DEstimator(CameraCalibration())
        
        # Automotive-specific class mapping
        self.automotive_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Performance tracking
        self.detection_times = []
        self.frame_count = 0
        
    def detect_objects(self, frame: np.ndarray) -> List[Detection3D]:
        """Enhanced object detection with 3D estimation"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections_3d = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter for automotive-relevant classes
                    if cls_id in self.automotive_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        class_name = self.automotive_classes[cls_id]
                        
                        # 3D estimation
                        distance = self.object_3d_estimator.estimate_distance(
                            (x1, y1, x2, y2), class_name
                        )
                        bbox_3d = self.object_3d_estimator.estimate_3d_bbox(
                            (x1, y1, x2, y2), class_name, distance
                        )
                        
                        # Calculate viewing angle
                        center_x = (x1 + x2) / 2
                        image_center = frame.shape[1] / 2
                        angle = math.atan2(center_x - image_center, 
                                         self.object_3d_estimator.camera.focal_length)
                        
                        # Basic risk scoring (enhanced in Week 4)
                        risk_score = self._calculate_basic_risk(distance, class_name)
                        
                        detection = Detection3D(
                            bbox_2d=(x1, y1, x2, y2),
                            bbox_3d=bbox_3d,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=class_name,
                            distance=distance,
                            velocity=(0.0, 0.0, 0.0),  # Will be calculated in Week 3
                            angle=angle,
                            risk_score=risk_score
                        )
                        
                        detections_3d.append(detection)
        
        # Performance tracking
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.frame_count += 1
        
        return detections_3d
    
    def _calculate_basic_risk(self, distance: float, object_type: str) -> float:
        """Basic risk calculation (will be enhanced in Week 4)"""
        # Risk increases as distance decreases
        base_risk = max(0, (50 - distance) / 50)  # Risk peaks at <50m
        
        # Different risk factors for different objects
        risk_multipliers = {
            'person': 1.5,      # Higher risk for pedestrians
            'bicycle': 1.3,     # Vulnerable road users
            'motorcycle': 1.2,
            'car': 1.0,
            'truck': 1.1,       # Larger vehicles
            'bus': 1.1
        }
        
        multiplier = risk_multipliers.get(object_type, 1.0)
        return min(base_risk * multiplier, 1.0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {}
            
        avg_time = np.mean(self.detection_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_detection_time': avg_time,
            'fps': fps,
            'total_frames': self.frame_count
        }
