from typing import Tuple
from camera_calibration import CameraCalibration

class Object3DEstimator:
    """Estimate 3D object properties from 2D detections"""
    
    def __init__(self, camera_calib: CameraCalibration):
        self.camera = camera_calib
        
        # Real-world object dimensions (average values in meters)
        self.object_dimensions = {
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'truck': {'width': 2.5, 'height': 3.0, 'length': 12.0},
            'bus': {'width': 2.5, 'height': 3.2, 'length': 12.0},
            'motorcycle': {'width': 0.8, 'height': 1.2, 'length': 2.0},
            'bicycle': {'width': 0.6, 'height': 1.0, 'length': 1.8},
            'person': {'width': 0.6, 'height': 1.7, 'length': 0.3}
        }
    
    def estimate_distance(self, bbox_2d: Tuple[int, int, int, int], 
                         object_type: str) -> float:
        """Estimate distance using object height in image"""
        x1, y1, x2, y2 = bbox_2d
        bbox_height_pixels = y2 - y1
        
        if object_type not in self.object_dimensions:
            object_type = 'car'  # Default assumption
            
        real_height = self.object_dimensions[object_type]['height']
        
        # Distance = (real_height * focal_length) / pixel_height
        distance = (real_height * self.camera.focal_length) / bbox_height_pixels
        return max(distance, 1.0)  # Minimum 1 meter
    
    def estimate_3d_position(self, bbox_2d: Tuple[int, int, int, int], 
                           distance: float) -> Tuple[float, float, float]:
        """Convert 2D bbox to 3D world coordinates"""
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Convert image coordinates to world coordinates
        # X: left-right (positive = right)
        # Y: up-down (positive = up)  
        # Z: forward-backward (positive = forward)
        
        world_x = (center_x - self.camera.cx) * distance / self.camera.focal_length
        world_y = self.camera.camera_height - (center_y - self.camera.cy) * distance / self.camera.focal_length
        world_z = distance
        
        return (world_x, world_y, world_z)
    
    def estimate_3d_bbox(self, bbox_2d: Tuple[int, int, int, int], 
                        object_type: str, distance: float) -> Tuple[float, float, float, float, float, float]:
        """Estimate 3D bounding box dimensions"""
        if object_type not in self.object_dimensions:
            object_type = 'car'
            
        dims = self.object_dimensions[object_type]
        x, y, z = self.estimate_3d_position(bbox_2d, distance)
        
        return (x, y, z, dims['width'], dims['height'], dims['length'])
