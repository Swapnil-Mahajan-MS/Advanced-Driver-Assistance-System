from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection3D:
    """Enhanced 3D detection with real-world coordinates"""
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_3d: Tuple[float, float, float, float, float, float]  # x, y, z, w, h, l
    confidence: float
    class_id: int
    class_name: str
    distance: float
    velocity: Tuple[float, float, float]  # vx, vy, vz
    angle: float  # viewing angle
    risk_score: float
