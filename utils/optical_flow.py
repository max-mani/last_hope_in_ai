import cv2
import numpy as np

def compute_optical_flow(prev_gray, curr_gray, scale=1.0):
    """
    Computes dense Farneback optical flow.

    `scale` (0 < scale <= 1.0): when < 1.0, Farneback runs on a downsampled
    copy of both frames (much cheaper — dense flow cost scales with pixel
    count) and the resulting flow field is resized back up to the original
    resolution before being returned, with vector magnitudes rescaled to
    compensate. Every downstream caller (get_mean_flow_magnitude,
    calculate_flow_angular_dispersion) samples the returned flow array in
    original-frame pixel coordinates exactly as before — this is purely a
    speed optimization, not an API change. scale=1.0 reproduces the
    original full-resolution behavior exactly.
    """
    if prev_gray is None or curr_gray is None:
        return None

    if scale is not None and scale < 1.0:
        h, w = curr_gray.shape[:2]
        small_w = max(2, int(round(w * scale)))
        small_h = max(2, int(round(h * scale)))
        prev_small = cv2.resize(prev_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        flow_small = cv2.calcOpticalFlowFarneback(
            prev_small,
            curr_small,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        # Displacement was measured in the downsampled grid — rescale vector
        # magnitudes back to original-resolution pixel units.
        flow[..., 0] /= scale
        flow[..., 1] /= scale
        return flow

    # Farneback parameters (full resolution — original behavior)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, 
        curr_gray, 
        None, 
        pyr_scale=0.5, 
        levels=3, 
        winsize=15, 
        iterations=3, 
        poly_n=5, 
        poly_sigma=1.2, 
        flags=0
    )
    return flow

def get_mean_flow_magnitude(flow, bbox):
    """
    Calculates the mean optical flow magnitude inside a bounding box.
    bbox: List or tuple (x1, y1, x2, y2)
    """
    if flow is None:
        return 0.0
    
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clip coordinates to image boundary
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    # Check if crop area is valid
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    flow_crop = flow[y1:y2, x1:x2]
    
    # Magnitude is sqrt(dx^2 + dy^2)
    magnitude = np.sqrt(flow_crop[..., 0]**2 + flow_crop[..., 1]**2)
    return float(np.mean(magnitude))

def calculate_flow_angular_dispersion(flow, bbox, min_magnitude=0.5):
    """
    Calculates the directional variance (dispersion) of flow angles within a bbox.
    Returns value between 0.0 (perfect parallel flow) and 1.0 (totally chaotic/dispersed flow).
    """
    if flow is None:
        return 0.0

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    flow_crop = flow[y1:y2, x1:x2]
    
    dx = flow_crop[..., 0]
    dy = flow_crop[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Filter out stationary or extremely slow movement to reduce noise
    valid_mask = magnitude > min_magnitude
    if not np.any(valid_mask):
        return 0.0
        
    dx_val = dx[valid_mask]
    dy_val = dy[valid_mask]
    
    angles = np.arctan2(dy_val, dx_val)
    
    # Directional stats
    cos_mean = np.mean(np.cos(angles))
    sin_mean = np.mean(np.sin(angles))
    
    R = np.sqrt(cos_mean**2 + sin_mean**2)
    # Circular variance
    return float(1.0 - R)

def calculate_frame_diff_ratio(prev_gray, curr_gray, bbox, threshold=20):
    """
    Computes the percentage of changed pixels (visual burst) inside a bbox
    between successive frames.
    """
    if prev_gray is None or curr_gray is None:
        return 0.0
        
    h, w = prev_gray.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    # Crop frames first to save computation
    prev_crop = prev_gray[y1:y2, x1:x2]
    curr_crop = curr_gray[y1:y2, x1:x2]
    
    # Absolute difference
    diff = cv2.absdiff(prev_crop, curr_crop)
    
    # Threshold the difference image
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    total_pixels = thresh.size
    if total_pixels <= 0:
        return 0.0
        
    changed_pixels = np.count_nonzero(thresh)
    return float(changed_pixels / total_pixels)
