import cv2
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
TARGET_LOCKED_THRESHOLD = 5  # Reduced frames to confirm target lock
CONFIDENCE_THRESHOLD = 0.35

# Display settings
DISPLAY_WIDTH = 640  # Reduced resolution for better performance
DISPLAY_HEIGHT = 480
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
        self.target_distance = 1.0  # Default distance (no depth estimation)
        self.target_history = []  # For smoothing
        self.last_detection_time = 0
        
tracking = TrackingState()

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
    center_threshold = 8  # Increased threshold for easier locking
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

def load_face_detector():
    """Load face detection cascade classifier"""
    try:
        # Load Face detector - using LBP for better performance
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        return face_cascade
    except Exception as e:
        print(f"Error loading face detector: {str(e)}")
        exit(1)

def draw_targeting_interface(frame, target_data=None):
    """Draw the heads-up display interface"""
    h, w = frame.shape[:2]
    
    # Use direct drawing instead of creating an overlay for better performance
    # Draw center crosshairs
    cv2.line(frame, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), (FRAME_CENTER_X + 20, FRAME_CENTER_Y), COLOR_WHITE, 1)
    cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), (FRAME_CENTER_X, FRAME_CENTER_Y + 20), COLOR_WHITE, 1)
    
    # Draw corner elements (simplified)
    # Top-left corner
    cv2.line(frame, (20, 20), (70, 20), COLOR_WHITE, 2)
    cv2.line(frame, (20, 20), (20, 70), COLOR_WHITE, 2)
    
    # Top-right corner
    cv2.line(frame, (w-20, 20), (w-70, 20), COLOR_WHITE, 2)
    cv2.line(frame, (w-20, 20), (w-20, 70), COLOR_WHITE, 2)
    
    # Display tracking state
    if tracking.locked:
        status_text = "FACE LOCKED"
        status_color = COLOR_RED
    else:
        status_text = "SCANNING"
        status_color = COLOR_GREEN
    
    cv2.putText(frame, status_text, (FRAME_CENTER_X - 80, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # If we have target data, draw targeting elements
    if target_data:
        # Draw connecting line to target
        cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 
                (target_data["center_x"], target_data["center_y"]), COLOR_GREEN, 1)
        
        # Draw target box
        if tracking.locked:
            box_color = COLOR_RED
            
            # Add animated elements for locked target (simplified)
            thickness = 2
            size = max(20, min(target_data["width"], target_data["height"]) // 2)
            
            # Draw corner indicators
            x, y = target_data["center_x"], target_data["center_y"]
            cv2.line(frame, (x - size, y - size), (x - size + 10, y - size), box_color, thickness)
            cv2.line(frame, (x - size, y - size), (x - size, y - size + 10), box_color, thickness)
            
            cv2.line(frame, (x + size, y - size), (x + size - 10, y - size), box_color, thickness)
            cv2.line(frame, (x + size, y - size), (x + size, y - size + 10), box_color, thickness)
            
            cv2.line(frame, (x - size, y + size), (x - size + 10, y + size), box_color, thickness)
            cv2.line(frame, (x - size, y + size), (x - size, y + size - 10), box_color, thickness)
            
            cv2.line(frame, (x + size, y + size), (x + size - 10, y + size), box_color, thickness)
            cv2.line(frame, (x + size, y + size), (x + size, y + size - 10), box_color, thickness)
        else:
            box_color = COLOR_YELLOW
            # Draw rectangle around target
            cv2.rectangle(frame, (target_data["center_x"] - target_data["width"]//2, 
                                target_data["center_y"] - target_data["height"]//2),
                        (target_data["center_x"] + target_data["width"]//2, 
                        target_data["center_y"] + target_data["height"]//2), 
                        box_color, 1)
        
        # Display offset information (simplified)
        x_text = f"X: {target_data['x_offset_percent']:+}%"
        y_text = f"Y: {target_data['y_offset_percent']:+}%"
        
        cv2.putText(frame, "FACE DETECTED", (w - 200, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        cv2.putText(frame, x_text, (w - 200, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        cv2.putText(frame, y_text, (w - 200, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, timestamp, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Add mode indicator
    cv2.putText(frame, "MODE: FACE TRACK", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    return frame

def main():
    # Load face detector
    face_cascade = load_face_detector()
    
    # Initialize video capture
    try:
        # Try to use webcam
        cap = cv2.VideoCapture(0)  # Try 0, 1, 2 for different cameras
        if not cap.isOpened():
            raise Exception("Could not access camera")
            
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        processing_time = 0
        
        print(f"Face tracking system initialized. Press 'q' to exit.")
        
        while True:
            # Start frame timing
            frame_start = time.time()
            
            # Get frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera")
                break
            
            # Process every other frame for better performance at high FPS
            if frame_count % 2 == 0:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use cascade classifier with optimized parameters
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,  
                    minNeighbors=5,   # Higher value for fewer false positives
                    minSize=(50, 50), # Minimum face size to detect
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process face detections
                all_faces = []
                
                for (x, y, w, h) in faces:
                    # Create face box
                    x1, y1 = x, y
                    x2, y2 = x+w, y+h  
                    
                    # Skip small detections
                    if w * h < 1000:  # Minimum area threshold
                        continue
                    
                    # Calculate targeting data directly
                    target_info = calculate_centering_data((x1, y1, x2, y2), frame_width, frame_height)
                    all_faces.append(target_info)
                
                # Update tracking state
                current_time = time.time()
                target_data = None
                
                # Select best target - largest face (closest to camera)
                if all_faces:
                    # Reset timeout counter
                    tracking.last_detection_time = current_time
                    
                    # Sort faces by size (area) and choose the largest
                    all_faces.sort(key=lambda x: x["area"], reverse=True)
                    best_target = all_faces[0]
                    
                    # Smooth tracking with history (using a smaller history)
                    if not tracking.target_history:
                        tracking.target_history = [best_target for _ in range(3)]
                    else:
                        tracking.target_history.pop(0)
                        tracking.target_history.append(best_target)
                    
                    # Average position for smoother tracking
                    avg_x = sum(t["center_x"] for t in tracking.target_history) // len(tracking.target_history)
                    avg_y = sum(t["center_y"] for t in tracking.target_history) // len(tracking.target_history)
                    avg_w = sum(t["width"] for t in tracking.target_history) // len(tracking.target_history)
                    avg_h = sum(t["height"] for t in tracking.target_history) // len(tracking.target_history)
                    
                    # Create smoothed target data
                    target_data = best_target.copy()
                    target_data["center_x"] = avg_x
                    target_data["center_y"] = avg_y
                    target_data["width"] = avg_w
                    target_data["height"] = avg_h
                    
                    # Only print occasionally to reduce CPU load
                    if frame_count % 15 == 0:
                        print(f"Face detected at X: {target_data['x_offset_percent']:+3d}%, "
                            f"Y: {target_data['y_offset_percent']:+3d}%, "
                            f"Size: {target_data['width']}x{target_data['height']}")
                    
                    # Update lock status
                    if best_target["is_centered"]:
                        tracking.lock_counter += 1
                        if tracking.lock_counter >= TARGET_LOCKED_THRESHOLD:
                            if not tracking.locked:
                                print("FACE LOCKED!")
                            tracking.locked = True
                    else:
                        tracking.lock_counter = max(0, tracking.lock_counter - 1)
                        if tracking.lock_counter == 0 and tracking.locked:
                            tracking.locked = False
                            print("Lock lost")
                else:
                    # No valid targets found
                    if current_time - tracking.last_detection_time > 0.5:  # 0.5 second timeout (faster response)
                        if tracking.locked:
                            print("No face detected. Lock released.")
                        tracking.locked = False
                        tracking.lock_counter = 0
                        tracking.target_history = []
            
            # Draw the interface (every frame)
            display_frame = draw_targeting_interface(frame, target_data if 'target_data' in locals() else None)
            
            # Calculate FPS every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / max(0.001, (end_time - start_time))
                processing_time = (end_time - frame_start) * 1000  # in ms
                start_time = end_time
            
            # Add FPS counter in the corner
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (frame_width - 120, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            
            # Display frame
            cv2.imshow("Face Tracking System", display_frame)
            
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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()