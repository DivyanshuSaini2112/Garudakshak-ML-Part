import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
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

# Configuration
TARGET_OBJECTS = ["airplane", "drone", "bird"]
CONFIDENCE_THRESHOLD = 0.35
TARGET_LOCKED_THRESHOLD = 10  # Frames to confirm target lock

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FRAME_CENTER_X = DISPLAY_WIDTH // 2
FRAME_CENTER_Y = DISPLAY_HEIGHT // 2

# Colors (BGR format)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# Tracking state
class TrackingState:
    def __init__(self):
        self.locked = False
        self.lock_counter = 0
        self.current_target = None
        self.target_distance = 0
        self.target_history = []  # For smoothing
        self.last_detection_time = 0
        
tracking = TrackingState()

def classify_flying_object(object_type, box_width, box_height, frame_width, frame_height):
    """Determine object classification based on size and detection"""
    object_type_lower = object_type.lower()
    relative_size = (box_width * box_height) / (frame_width * frame_height)
    
    if object_type_lower == "airplane" and relative_size < 0.15:
        return "drone"
    elif object_type_lower == "bird" and relative_size < 0.05:
        return "drone"
    return object_type

def calculate_centering_data(box, frame_width, frame_height):
    """Calculate targeting data for the object"""
    x1, y1, x2, y2 = box
    
    # Calculate center and dimensions
    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Calculate distance from center (as percentage)
    x_offset = object_center_x - FRAME_CENTER_X
    y_offset = object_center_y - FRAME_CENTER_Y
    x_offset_percent = int(x_offset / (frame_width / 2) * 100)
    y_offset_percent = int(y_offset / (frame_height / 2) * 100)
    
    # Calculate diagonal distance for lock-on indicator
    diagonal_distance = np.sqrt(x_offset**2 + y_offset**2)
    diagonal_distance_normalized = diagonal_distance / (np.sqrt(frame_width**2 + frame_height**2) / 2)
    
    # Target locked status
    center_threshold = 5
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
    """Load AI models for detection and depth estimation"""
    try:
        # Load YOLO model for object detection
        model = YOLO("yolov8x.pt")
        model.to(device)
        
        # Load MiDaS for depth estimation
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device).eval()
        
        # Use full precision for better accuracy
        if device.type == 'cuda':
            model.model.float()
            midas.float()
            
        return model, midas
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Required packages: torch torchvision ultralytics opencv-python pillow timm")
        exit(1)

# Transform for depth estimation
transform = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Thread-local storage
thread_local = threading.local()

def get_thread_local_tensor():
    """Get reusable tensor for depth processing"""
    if not hasattr(thread_local, 'tensor'):
        thread_local.tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
    return thread_local.tensor

@torch.inference_mode()
def estimate_depth(frame, box):
    """Estimate depth of object in frame"""
    try:
        x1, y1, x2, y2 = box
        object_image = frame[y1:y2, x1:x2]
        
        # Skip if the object is too small
        if object_image.shape[0] < 10 or object_image.shape[1] < 10:
            return 1.0

        # Process image for depth estimation
        object_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
        
        input_tensor = get_thread_local_tensor()
        with torch.no_grad():
            input_data = transform(object_image).to(device, dtype=torch.float32, non_blocking=True)
            input_tensor[0] = input_data
            
            depth_map = midas(input_tensor)
            average_depth = torch.mean(depth_map).item()
        
        # Scale depth to a more intuitive range (0-1000m)
        scaled_depth = min(max(1.0, average_depth * 150), 1000)
        return scaled_depth
    except Exception as e:
        print(f"Depth estimation error: {str(e)}")
        return 1.0

def draw_targeting_interface(frame, target_data=None):
    """Draw the heads-up display interface"""
    h, w = frame.shape[:2]
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    
    # Draw center crosshairs
    cv2.line(overlay, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), (FRAME_CENTER_X + 20, FRAME_CENTER_Y), COLOR_WHITE, 1)
    cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), (FRAME_CENTER_X, FRAME_CENTER_Y + 20), COLOR_WHITE, 1)
    
    # Draw corner elements (similar to video)
    # Top-left corner
    cv2.line(overlay, (20, 20), (70, 20), COLOR_WHITE, 2)
    cv2.line(overlay, (20, 20), (20, 70), COLOR_WHITE, 2)
    
    # Top-right corner
    cv2.line(overlay, (w-20, 20), (w-70, 20), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, 20), (w-20, 70), COLOR_WHITE, 2)
    
    # Bottom-left corner
    cv2.line(overlay, (20, h-20), (70, h-20), COLOR_WHITE, 2)
    cv2.line(overlay, (20, h-20), (20, h-70), COLOR_WHITE, 2)
    
    # Bottom-right corner
    cv2.line(overlay, (w-20, h-20), (w-70, h-20), COLOR_WHITE, 2)
    cv2.line(overlay, (w-20, h-20), (w-20, h-70), COLOR_WHITE, 2)
    
    # Display tracking state
    if tracking.locked:
        status_text = "TARGET LOCKED"
        status_color = COLOR_RED
    else:
        status_text = "SCANNING"
        status_color = COLOR_GREEN
    
    cv2.putText(overlay, status_text, (FRAME_CENTER_X - 80, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # If we have target data, draw targeting elements
    if target_data:
        # Draw connecting line to target
        cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y), 
                (target_data["center_x"], target_data["center_y"]), COLOR_GREEN, 1)
        
        # Draw target box
        if tracking.locked:
            box_color = COLOR_RED
            
            # Add animated elements for locked target
            thickness = 2
            size = max(30, min(target_data["width"], target_data["height"]) // 2)
            
            # Draw corner indicators
            x, y = target_data["center_x"], target_data["center_y"]
            cv2.line(overlay, (x - size, y - size), (x - size + 10, y - size), box_color, thickness)
            cv2.line(overlay, (x - size, y - size), (x - size, y - size + 10), box_color, thickness)
            
            cv2.line(overlay, (x + size, y - size), (x + size - 10, y - size), box_color, thickness)
            cv2.line(overlay, (x + size, y - size), (x + size, y - size + 10), box_color, thickness)
            
            cv2.line(overlay, (x - size, y + size), (x - size + 10, y + size), box_color, thickness)
            cv2.line(overlay, (x - size, y + size), (x - size, y + size - 10), box_color, thickness)
            
            cv2.line(overlay, (x + size, y + size), (x + size - 10, y + size), box_color, thickness)
            cv2.line(overlay, (x + size, y + size), (x + size, y + size - 10), box_color, thickness)
        else:
            box_color = COLOR_YELLOW
            # Draw rectangle around target
            cv2.rectangle(overlay, (target_data["center_x"] - target_data["width"]//2, 
                                   target_data["center_y"] - target_data["height"]//2),
                         (target_data["center_x"] + target_data["width"]//2, 
                          target_data["center_y"] + target_data["height"]//2), 
                         box_color, 1)
        
        # Display distance and offset information
        distance_text = f"DIST: {tracking.target_distance:.1f}m"
        x_text = f"X: {target_data['x_offset_percent']:+}%"
        y_text = f"Y: {target_data['y_offset_percent']:+}%"
        
        cv2.putText(overlay, distance_text, (w - 250, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        cv2.putText(overlay, x_text, (w - 250, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        cv2.putText(overlay, y_text, (w - 250, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    date_str = time.strftime("%Y-%m-%d", time.localtime())
    cv2.putText(overlay, timestamp, (20, h - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    cv2.putText(overlay, date_str, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Add mode indicator
    cv2.putText(overlay, "MODE: AUTO", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Blend overlay with original frame
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def main():
    # Load models
    model, midas = load_models()
    
    # Initialize video capture
    try:
        # Try to use webcam
        cap = cv2.VideoCapture(0)  # Try 0, 1, 2 for different cameras
        if not cap.isOpened():
            raise Exception("Could not access camera")
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up output video
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"drone_tracking_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
        
        # Set up processing threads
        executor = ThreadPoolExecutor(max_workers=4)
        
        # Create a buffer for processing
        frame_buffer = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        print(f"Tracking system initialized. Press 'q' to exit.")
        print(f"Output video will be saved to: {output_path}")
        
        while True:
            # Get frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera")
                break
                
            # Copy frame to buffer for processing
            np.copyto(frame_buffer, frame)
            
            # Run YOLO detection
            results = model(frame_buffer, conf=CONFIDENCE_THRESHOLD, iou=0.45, max_det=10)
            
            # Process detection results
            futures = []
            valid_targets = []
            
            for result in results:
                for box in result.boxes:
                    # Get bounding box and object info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    object_type = model.names[int(box.cls[0])]
                    
                    # Skip small detections
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width * box_height < 400:  # Minimum area threshold
                        continue
                    
                    # Refine classification
                    object_type = classify_flying_object(object_type, box_width, box_height, 
                                                       frame_width, frame_height)
                    
                    # Check if this is a target we're interested in
                    if object_type.lower() not in [t.lower() for t in TARGET_OBJECTS]:
                        continue
                        
                    # Start depth estimation
                    future = executor.submit(estimate_depth, frame_buffer, (x1, y1, x2, y2))
                    futures.append((future, (x1, y1, x2, y2), object_type, confidence))
            
            # Update tracking state
            current_time = time.time()
            target_data = None
            
            # Process results and find best target
            for future, box, object_type, confidence in futures:
                depth = future.result()
                
                # Calculate targeting data
                target_info = calculate_centering_data(box, frame_width, frame_height)
                target_info["depth"] = depth
                target_info["confidence"] = confidence
                target_info["type"] = object_type
                
                valid_targets.append(target_info)
            
            # Select best target (prioritize centered and large objects)
            if valid_targets:
                # Reset timeout counter
                tracking.last_detection_time = current_time
                
                # Sort by combination of size and center distance
                valid_targets.sort(key=lambda x: (x["area"] * (1.0 - x["diagonal_distance"])), reverse=True)
                best_target = valid_targets[0]
                
                # Smooth tracking with history
                if not tracking.target_history:
                    tracking.target_history = [best_target for _ in range(5)]
                else:
                    tracking.target_history.pop(0)
                    tracking.target_history.append(best_target)
                
                # Average position and depth for smoother tracking
                avg_x = sum(t["center_x"] for t in tracking.target_history) // len(tracking.target_history)
                avg_y = sum(t["center_y"] for t in tracking.target_history) // len(tracking.target_history)
                avg_w = sum(t["width"] for t in tracking.target_history) // len(tracking.target_history)
                avg_h = sum(t["height"] for t in tracking.target_history) // len(tracking.target_history)
                avg_depth = sum(t["depth"] for t in tracking.target_history) / len(tracking.target_history)
                
                # Create smoothed target data
                target_data = best_target.copy()
                target_data["center_x"] = avg_x
                target_data["center_y"] = avg_y
                target_data["width"] = avg_w
                target_data["height"] = avg_h
                
                # Store distance
                tracking.target_distance = avg_depth
                
                # Update lock status
                if best_target["is_centered"]:
                    tracking.lock_counter += 1
                    if tracking.lock_counter >= TARGET_LOCKED_THRESHOLD:
                        tracking.locked = True
                else:
                    tracking.lock_counter = max(0, tracking.lock_counter - 1)
                    if tracking.lock_counter == 0:
                        tracking.locked = False
            else:
                # No valid targets found
                if current_time - tracking.last_detection_time > 1.0:  # 1 second timeout
                    tracking.locked = False
                    tracking.lock_counter = 0
                    tracking.target_history = []
            
            # Draw the interface
            display_frame = draw_targeting_interface(frame, target_data)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
            
            # Add FPS counter in the corner
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (frame_width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
            
            # Display and record
            cv2.imshow("Advanced Drone Targeting System", display_frame)
            out.write(display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        print("Shutting down system...")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        if 'executor' in locals():
            executor.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()