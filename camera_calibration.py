import numpy as np

class CameraCalibration:
    """Camera intrinsic parameters for 3D reconstruction"""
    def __init__(self, focal_length=800, image_width=1280, image_height=720):
        self.focal_length = focal_length
        self.cx = image_width / 2  # Principal point x
        self.cy = image_height / 2  # Principal point y
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera matrix
        self.K = np.array([
            [focal_length, 0, self.cx],
            [0, focal_length, self.cy],
            [0, 0, 1]
        ])
        
        # Vehicle-specific parameters
        self.camera_height = 1.5  # meters above ground
        self.camera_pitch = 0.1  # radians (slight downward tilt)
