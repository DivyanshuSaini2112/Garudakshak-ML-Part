import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import logging

# GPIO Setup for Servo Motor
SERVO_TRACKING_PIN = 18  # GPIO 18 for tracking servo motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_TRACKING_PIN, GPIO.OUT)
tracking_servo = GPIO.PWM(SERVO_TRACKING_PIN, 50)  # 50 Hz PWM
tracking_servo.start(7.5)  # Neutral position (90 degrees)

# Configuration
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FRAME_CENTER_X = DISPLAY_WIDTH // 2
FRAME_CENTER_Y = DISPLAY_HEIGHT // 2

# Colors (BGR format)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

class TrackingState:
    def __init__(self):
        self.current_servo_angle = 90  # Initial servo position (degrees)
        self.last_detection_time = 0
        self.face_detected = False

tracking = TrackingState()

def load_face_detector():
    """Load face detection models with comprehensive error handling"""
    # Setup logging for this function
    logger = logging.getLogger(__name__)
    
    # Check for Haar Cascade
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    if not os.path.exists(haar_cascade_path):
        logger.error(f"Haar Cascade file not found at {haar_cascade_path}")
        return None, None

    haar_cascade = cv2.CascadeClassifier(haar_cascade_path)
    
    # Check for DNN model files
    prototxt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    
    # DNN is optional, so we'll continue even if it's not found
    dnn_net = None
    if os.path.exists(prototxt_path) and os.path.exists(model_path):
        try:
            dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            logger.info("Successfully loaded DNN face detector")
        except Exception as e:
            logger.warning(f"Error loading DNN face detector: {str(e)}")
    else:
        logger.warning("DNN face detector files not found. Falling back to Haar Cascade.")
    
    return haar_cascade, dnn_net

def move_tracking_servo(angle):
    """Move servo to specific angle with error handling"""
    try:
        # Convert angle (0-180) to duty cycle (2.5-12.5)
        duty = 2.5 + (angle / 18.0)  # Linear mapping from 0-180 to 2.5-12.5
        tracking_servo.ChangeDutyCycle(duty)
        time.sleep(0.05)  # Allow servo to move
        tracking_servo.ChangeDutyCycle(0)  # Stop signal to prevent jitter
    except Exception as e:
        logging.error(f"Servo movement error: {e}")

def calculate_face_offset(box, frame_width, frame_height):
    """Calculate horizontal offset of face from frame center"""
    x1, y1, x2, y2 = box
    object_center_x = (x1 + x2) // 2
    x_offset = object_center_x - FRAME_CENTER_X
    x_offset_percent = int(x_offset / (frame_width / 2) * 100)
    return x_offset_percent

def control_tracking_motor(x_offset_percent):
    """Control servo based on face position"""
    # Tracking motor control parameters
    sensitivity = 1.5  # Degrees per percent offset
    threshold = 10   # Minimum offset to trigger movement
    
    if abs(x_offset_percent) > threshold:
        # Calculate new angle
        adjustment = x_offset_percent * sensitivity
        new_angle = tracking.current_servo_angle - adjustment  # Invert for correct direction
        new_angle = max(0, min(180, new_angle))  # Clamp to servo range
        
        tracking.current_servo_angle = new_angle
        move_tracking_servo(new_angle)
        logging.info(f"Tracking servo moved to {new_angle:.1f} degrees")

def detect_faces(frame, haar_cascade, dnn_net):
    """Detect faces using multiple methods"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = frame.shape[:2]
    
    # Haar Cascade Detection (Primary Method)
    haar_faces = haar_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),  # Reduced minimum size
        maxSize=(300, 300)  # Added maximum size
    )
    
    # Combine Haar faces
    combined_faces = []
    for (x, y, w, h) in haar_faces:
        if w * h >= 100:  # Minimum face size
            combined_faces.append((x, y, x+w, y+h))
    
    # If DNN net is available, use as secondary method
    if dnn_net is not None:
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            
            # Process DNN Detections
            confidence_threshold = 0.5
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    x1, y1, x2, y2 = box.astype("int")
                    
                    # Filter small or invalid faces
                    if x2 - x1 > 30 and y2 - y1 > 30:
                        combined_faces.append((x1, y1, x2, y2))
        except Exception as e:
            logging.error(f"DNN face detection error: {e}")
    
    return combined_faces

def draw_interface(frame, faces=None):
    """Draw tracking interface on frame"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw crosshair
    cv2.line(overlay, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), 
             (FRAME_CENTER_X + 20, FRAME_CENTER_Y), COLOR_WHITE, 1)
    cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), 
             (FRAME_CENTER_X, FRAME_CENTER_Y + 20), COLOR_WHITE, 1)
    
    # Status text
    status_text = "FACE TRACKING" if faces else "NO FACE"
    status_color = COLOR_GREEN if faces else COLOR_RED
    cv2.putText(overlay, status_text, (FRAME_CENTER_X - 80, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Draw faces and info
    if faces:
        for face in faces:
            cv2.rectangle(overlay, (face[0], face[1]), 
                          (face[2], face[3]), COLOR_YELLOW, 2)
        
        # Servo and offset information
        info_texts = [
            f"Detected Faces: {len(faces)}",
            f"Servo Angle: {tracking.current_servo_angle:.1f}°"
        ]
        for i, text in enumerate(info_texts):
            cv2.putText(overlay, text, (w - 200, h - 100 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Timestamp
    timestamp = time.strftime("%H:%M:%S %Y-%m-%d", time.localtime())
    cv2.putText(overlay, timestamp, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return overlay

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler("face_tracking.log"),
            logging.StreamHandler()
        ]
    )
    
    # Load face detectors
    haar_cascade, dnn_net = load_face_detector()
    
    if haar_cascade is None:
        logging.error("No face detection method available. Exiting.")
        return
    
    # Open video capture with multiple camera indices
    camera_indices = [0, 1, 2]  # Try these camera indices
    cap = None
    
    for index in camera_indices:
        try:
            cap = cv2.VideoCapture(index)
            
            # Additional camera capability checks
            if cap.isOpened():
                # Test frame capture
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    logging.info(f"Successfully opened camera at index {index}")
                    break
                else:
                    logging.warning(f"Camera at index {index} failed frame capture test")
            
            cap.release()
        except Exception as e:
            logging.error(f"Error testing camera at index {index}: {e}")
    
    if cap is None or not cap.isOpened():
        logging.critical("Error: Could not open any camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    logging.info("Face Tracking System Initialized. Press 'q' to quit.")
    
    try:
        frame_count = 0
        detection_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logging.warning("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detect_faces(frame, haar_cascade, dnn_net)
            
            # Process largest face for tracking
            if faces:
                detection_count += 1
                # Sort faces by size and select largest
                largest_face = max(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]))
                
                # Calculate offset and control servo
                x_offset_percent = calculate_face_offset(largest_face, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                control_tracking_motor(x_offset_percent)
            
            # Draw interface
            processed_frame = draw_interface(frame, faces)
            
            # Display frame
            cv2.imshow('Face Tracking System', processed_frame)
            
            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except Exception as e:
        logging.error(f"Error in tracking loop: {e}", exc_info=True)
    
    finally:
        # Detailed cleanup and logging
        logging.info(f"Total frames processed: {frame_count}")
        logging.info(f"Faces detected: {detection_count}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracking_servo.stop()
        GPIO.cleanup()
        logging.info("Face Tracking System Terminated.")

if __name__ == "__main__":
    main()