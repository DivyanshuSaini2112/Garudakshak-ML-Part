import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

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
    try:
        # Multiple face detection methods
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Deep learning face detector
        prototxt_path = 'deploy.prototxt'
        model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
        dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        return haar_cascade, dnn_net
    except Exception as e:
        print(f"Error loading face detectors: {str(e)}")
        return None, None

def move_tracking_servo(angle):
    """Move servo to specific angle"""
    try:
        # Convert angle (0-180) to duty cycle (2.5-12.5)
        duty = 2.5 + (angle / 18.0)  # Linear mapping from 0-180 to 2.5-12.5
        tracking_servo.ChangeDutyCycle(duty)
        time.sleep(0.05)  # Allow servo to move
        tracking_servo.ChangeDutyCycle(0)  # Stop signal to prevent jitter
    except Exception as e:
        print(f"Servo movement error: {e}")

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
        print(f"Tracking servo moved to {new_angle:.1f} degrees")

def detect_faces(frame, haar_cascade, dnn_net):
    """Detect faces using multiple methods"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = frame.shape[:2]
    
    # Haar Cascade Detection
    haar_faces = haar_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50)
    )
    
    # Deep Learning Face Detector
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    
    # Combine detections
    combined_faces = []
    
    # Process Haar Cascade faces
    for (x, y, w, h) in haar_faces:
        if w * h >= 500:  # Minimum face size
            combined_faces.append((x, y, x+w, y+h))
    
    # Process DNN Detections
    confidence_threshold = 0.5
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            x1, y1, x2, y2 = box.astype("int")
            
            # Filter small or invalid faces
            if x2 - x1 > 50 and y2 - y1 > 50:
                combined_faces.append((x1, y1, x2, y2))
    
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
    # Load face detectors
    haar_cascade, dnn_net = load_face_detector()
    
    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Face Tracking System Initialized. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = detect_faces(frame, haar_cascade, dnn_net)
            
            # Process largest face for tracking
            if faces:
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error in tracking loop: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracking_servo.stop()
        GPIO.cleanup()
        print("Face Tracking System Terminated.")

if __name__ == "__main__":
    main()