import warnings ;warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from geopy.distance import geodesic
from PIL import Image
import torch.backends.cudnn as cudnn
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os

# Optimize for performance
cudnn.benchmark = True
cudnn.deterministic = False
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPS Settings
CAMERA_GPS = (37.7749, -122.4194)
LANDMARKS = {
    "Zone A": (37.7755, -122.4185),
    "Zone B": (37.7740, -122.4200)
}

# Configuration
CONFIDENCE_THRESHOLD = 0.35
TARGET_LOCKED_THRESHOLD = 5
GPS_UPDATE_INTERVAL = 1.0

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FRAME_CENTER_X = DISPLAY_WIDTH // 2
FRAME_CENTER_Y = DISPLAY_HEIGHT // 2

# Colors (BGR format)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

class TrackingState:
    def __init__(self):
        self.locked = False
        self.lock_counter = 0
        self.current_target = "target"
        self.target_distance = 0
        self.target_history = []
        self.last_detection_time = 0
        self.target_gps = None
        self.last_gps_update = 0
        self.smoothed_gps = None
        self.last_position = None
        self.last_time = time.time()
        
tracking = TrackingState()

def calculate_centering_data(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2
    
    # Define box dimensions
    box_width = x2 - x1
    box_height = y2 - y1
    
    # X offset: Positive = right, Negative = left
    x_offset = object_center_x - FRAME_CENTER_X
    # Y offset: Positive = up (above center), Negative = down (below center)
    y_offset = FRAME_CENTER_Y - object_center_y
    
    x_offset_percent = int(x_offset / (frame_width / 2) * 100)
    y_offset_percent = int(y_offset / (frame_height / 2) * 100)
    
    diagonal_distance = np.sqrt(x_offset**2 + y_offset**2)
    diagonal_distance_normalized = diagonal_distance / (np.sqrt(frame_width**2 + frame_height**2) / 2)
    
    center_threshold = 8
    is_centered = (abs(x_offset_percent) < center_threshold and 
                  abs(y_offset_percent) < center_threshold)
    
    return {
        "center_x": object_center_x,
        "center_y": object_center_y,
        "width": box_width,
        "height": box_height,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "x_offset_percent": x_offset_percent,
        "y_offset_percent": y_offset_percent,
        "diagonal_distance": diagonal_distance_normalized,
        "is_centered": is_centered,
        "area": box_width * box_height
    }

def load_models():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device).eval()
        if device.type == 'cuda':
            midas.float()
        return face_cascade, midas
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Required packages: torch torchvision opencv-python pillow geopy")
        exit(1)

transform = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

thread_local = threading.local()

def get_thread_local_tensor():
    if not hasattr(thread_local, 'tensor'):
        thread_local.tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
    return thread_local.tensor

@torch.inference_mode()
def estimate_depth(frame, box, midas_model):
    try:
        x1, y1, x2, y2 = box
        object_image = frame[y1:y2, x1:x2]
        
        if object_image.shape[0] < 10 or object_image.shape[1] < 10:
            return 1.0

        object_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
        input_tensor = get_thread_local_tensor()
        
        with torch.no_grad():
            input_data = transform(object_image).to(device, dtype=torch.float32, non_blocking=True)
            input_tensor[0] = input_data
            depth_map = midas_model(input_tensor)
            average_depth = torch.mean(depth_map).item()
        
        scaled_depth = min(max(0.5, average_depth * 10), 10)
        return scaled_depth
    except Exception as e:
        print(f"Depth estimation error: {str(e)}")
        return 1.0

def triangulate_position(target_depth, target_data):
    try:
        camera_lat, camera_lon = CAMERA_GPS
        current_time = time.time()
        
        fov_rad = np.radians(60)
        x_offset_norm = target_data["x_offset_percent"] / 100.0
        y_offset_norm = target_data["y_offset_percent"] / 100.0
        bearing_rad = x_offset_norm * (fov_rad / 2)
        
        distance = target_depth * 10
        
        lat_change = (distance * np.cos(bearing_rad)) / 111320
        lon_change = (distance * np.sin(bearing_rad)) / (111320 * np.cos(np.radians(camera_lat)))
        
        new_gps = (camera_lat + lat_change, camera_lon + lon_change)
        
        if (tracking.last_gps_update == 0 or 
            (current_time - tracking.last_gps_update) > GPS_UPDATE_INTERVAL):
            
            if tracking.smoothed_gps is None:
                tracking.smoothed_gps = new_gps
            else:
                old_lat, old_lon = tracking.smoothed_gps
                new_lat = old_lat * 0.7 + new_gps[0] * 0.3
                new_lon = old_lon * 0.7 + new_gps[1] * 0.3
                tracking.smoothed_gps = (new_lat, new_lon)
            
            movement_noise_lat = np.random.normal(0, 0.000005)
            movement_noise_lon = np.random.normal(0, 0.000005)
            tracking.target_gps = (
                tracking.smoothed_gps[0] + movement_noise_lat,
                tracking.smoothed_gps[1] + movement_noise_lon
            )
            tracking.last_gps_update = current_time
        
        return tracking.target_gps or new_gps
    except Exception as e:
        print(f"Position calculation error: {str(e)}")
        return CAMERA_GPS

def draw_targeting_interface(frame, target_data=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.line(overlay, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), 
             (FRAME_CENTER_X + 20, FRAME_CENTER_Y), COLOR_WHITE, 1)
    cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), 
             (FRAME_CENTER_X, FRAME_CENTER_Y + 20), COLOR_WHITE, 1)
    
    cv2.line(overlay, (20, 20), (70, 20), COLOR_WHITE, 2)
    cv2.line(overlay, (20, 20), (20, 70), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, 20), (w-70, 20), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, 20), (w-20, 70), COLOR_WHITE, 2)
    cv2.line(overlay, (20, h-20), (70, h-20), COLOR_WHITE, 2)
    cv2.line(overlay, (20, h-20), (20, h-70), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, h-20), (w-70, h-20), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, h-20), (w-20, h-70), COLOR_WHITE, 2)
    
    status_text = "TARGET LOCKED" if tracking.locked else "SCANNING"
    status_color = COLOR_RED if tracking.locked else COLOR_GREEN
    cv2.putText(overlay, status_text, (FRAME_CENTER_X - 80, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    if target_data:
        cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y), 
                (target_data["center_x"], target_data["center_y"]), COLOR_GREEN, 1)
        
        if tracking.locked:
            box_color = COLOR_RED
            thickness = 2
            size = max(30, min(target_data["width"], target_data["height"]) // 2)
            x, y = target_data["center_x"], target_data["center_y"]
            for dx, dy in [(-size, -size), (size, -size), (-size, size), (size, size)]:
                cv2.line(overlay, (x + dx, y + dy), (x + dx + 10 * (1 if dx > 0 else -1), y + dy), box_color, thickness)
                cv2.line(overlay, (x + dx, y + dy), (x + dx, y + dy + 10 * (1 if dy > 0 else -1)), box_color, thickness)
        else:
            box_color = COLOR_YELLOW
            cv2.rectangle(overlay, (target_data["center_x"] - target_data["width"]//2, 
                                  target_data["center_y"] - target_data["height"]//2),
                         (target_data["center_x"] + target_data["width"]//2, 
                          target_data["center_y"] + target_data["height"]//2), 
                         box_color, 1)
        
        info_texts = [
            "TARGET DETECTED",
            f"DIST: {tracking.target_distance:.1f}m",
            f"X: {target_data['x_offset_percent']:+}%",
            f"Y: {target_data['y_offset_percent']:+}%"
        ]
        for i, text in enumerate(info_texts):
            cv2.putText(overlay, text, (w - 250, h - 150 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        if tracking.target_gps:
            gps_text = f"GPS: {tracking.target_gps[0]:.6f}, {tracking.target_gps[1]:.6f}"
            cv2.putText(overlay, gps_text, (w - 430, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    timestamp = time.strftime("%H:%M:%S %Y-%m-%d", time.localtime())
    cv2.putText(overlay, timestamp, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    cv2.putText(overlay, "MODE: ANTI-DRONE", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame

def process_frame(frame, face_detector, midas_model, executor):
    start_time = time.time()
    preprocess_start = time.time()
    
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocess_time = (time.time() - preprocess_start) * 1000
    
    inference_start = time.time()
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    inference_time = (time.time() - inference_start) * 1000
    
    postprocess_start = time.time()
    target_data = None
    current_time = time.time()
    num_targets = len(faces)
    
    # ANSI color codes
    GREEN = "\033[92m"  # Bright green
    RED = "\033[91m"    # Bright red
    RESET = "\033[0m"   # Reset color
    
    if num_targets > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        box = (x, y, x+w, y+h)
        
        if w * h >= 1000:
            target_data = calculate_centering_data(box, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            tracking.last_detection_time = current_time
            
            future = executor.submit(estimate_depth, frame, box, midas_model)
            try:
                tracking.target_distance = future.result(timeout=0.1)
            except:
                tracking.target_distance = 1.0
            
            tracking.target_gps = triangulate_position(tracking.target_distance, target_data)
            
            # Calculate speed based on position change
            if tracking.last_position and tracking.target_gps:
                elapsed_time = current_time - tracking.last_time
                dist = geodesic(tracking.last_position, tracking.target_gps).meters
                speed = dist / elapsed_time if elapsed_time > 0 else 0
            else:
                speed = 0
            tracking.last_position = tracking.target_gps
            
            # Corrected adjustment instructions
            gps_text = f"GPS: {tracking.target_gps[0]:.6f}, {tracking.target_gps[1]:.6f}" if tracking.target_gps else "GPS: None"
            x_instruction = f"Move {'left' if target_data['x_offset_percent'] > 0 else 'right'} {abs(target_data['x_offset_percent'])}%"
            y_instruction = f"Move {'up' if target_data['y_offset_percent'] < 0 else 'down'} {abs(target_data['y_offset_percent'])}%"
            print(f"Face detected at X: {target_data['x_offset_percent']:+3}%, Y: {target_data['y_offset_percent']:+3}%, Size: {target_data['width']}x{target_data['height']}")
            print(f"Position: {gps_text}, Adjustment: {x_instruction}, {y_instruction}")
            
            if target_data["is_centered"]:
                tracking.lock_counter = min(tracking.lock_counter + 1, TARGET_LOCKED_THRESHOLD)
                if tracking.lock_counter == TARGET_LOCKED_THRESHOLD:
                    print(f"{GREEN}🔒 DRONE LOCKED!{RESET}")
                    tracking.locked = True
            else:
                tracking.lock_counter = max(tracking.lock_counter - 1, 0)
                if tracking.lock_counter == 0 and tracking.locked:
                    print(f"{RED}🔓 Lock lost{RESET}")
                    tracking.locked = False
                elif tracking.locked:
                    print(f"{GREEN}🔒 DRONE LOCKED!{RESET}")  # Print every frame while locked
            
    else:
        speed = 0
        if current_time - tracking.last_detection_time > 0.5:
            if tracking.locked:
                print(f"{RED}🔓 Lock lost{RESET}")
            tracking.locked = False
            tracking.lock_counter = 0
            tracking.target_history = []
    
    processed_frame = draw_targeting_interface(frame, target_data)
    postprocess_time = (time.time() - postprocess_start) * 1000
    total_time = (time.time() - start_time) * 1000
    
    # Shortened timing output
    print(f"Timing: {total_time:.1f}ms (P:{preprocess_time:.1f}/I:{inference_time:.1f}/PP:{postprocess_time:.1f})\n")
    
    tracking.last_time = current_time
    return processed_frame

def main():
    face_detector, midas_model = load_models()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Anti-Drone Detection System initialized. Press 'q' to exit.")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                processed_frame = process_frame(frame, face_detector, midas_model, executor)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - start_time)
                    start_time = time.time()
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (DISPLAY_WIDTH - 120, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                
                cv2.imshow("Anti-Drone Detection System", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()