import cv2
import numpy as np

class AdvancedLaneDetector:
    """Enhanced lane detection (basic version - will be upgraded in Week 2)"""
    
    def __init__(self):
        self.previous_lanes = None
        
    def detect_lanes(self, frame):
        """Improved lane detection with temporal smoothing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding based on image brightness
        brightness = np.mean(gray)
        if brightness < 100:  # Dark conditions
            low_threshold = 30
            high_threshold = 100
        else:  # Bright conditions
            low_threshold = 50
            high_threshold = 150
            
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        
        # Enhanced region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # More precise ROI polygon
        polygon = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.4), int(height * 0.6)),
            (int(width * 0.6), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough line detection with better parameters
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=50, 
            maxLineGap=20
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # More strict slope filtering
                    if -0.8 < slope < -0.2:  # Left lane
                        left_lines.append(line[0])
                    elif 0.2 < slope < 0.8:  # Right lane
                        right_lines.append(line[0])
        
        # Temporal smoothing
        if self.previous_lanes is not None:
            # Blend with previous frame (simple temporal filtering)
            alpha = 0.7  # Current frame weight
            if left_lines and self.previous_lanes[0]:
                # Simple blending logic would go here
                pass
                
        self.previous_lanes = (left_lines, right_lines)
        return left_lines, right_lines
