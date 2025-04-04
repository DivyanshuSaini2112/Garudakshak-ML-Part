import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import picamera
import picamera.array

# GPIO Setup for Servo Motor
SERVO_TRACKING_PIN = 18  # GPIO 18 for tracking servo motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_TRACKING_PIN, GPIO.OUT)
tracking_servo = GPIO.PWM(SERVO_TRACKING_PIN, 50)  # 50 Hz PWM
tracking_servo.start(7.5)  # Neutral position (90 degrees)

# Configuration
CONFIDENCE_THRESHOLD = 0.5
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

def calculate_face_offset(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    object_center_x = (x1 + x2) // 2
    x_offset = object_center_x - FRAME_CENTER_X
    x_offset_percent = int(x_offset / (frame_width / 2) * 100)
    return x_offset_percent

def move_tracking_servo(angle):
    # Convert angle (0-180) to duty cycle (2.5-12.5)
    duty = 2.5 + (angle / 18.0)  # Linear mapping from 0-180 to 2.5-12.5
    tracking_servo.ChangeDutyCycle(duty)
    time.sleep(0.05)  # Allow servo to move
    tracking_servo.ChangeDutyCycle(0)  # Stop signal to prevent jitter

def control_tracking_motor(x_offset_percent):
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

def load_face_detector():
    try:
        # Haar Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        return face_cascade
    except Exception as e:
        print(f"Error loading face detector: {str(e)}")
        exit(1)

def draw_targeting_interface(frame, face_data=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw crosshair
    cv2.line(overlay, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), 
             (FRAME_CENTER_X + 20, FRAME_CENTER_Y), COLOR_WHITE, 1)
    cv2.line(overlay, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), 
             (FRAME_CENTER_X, FRAME_CENTER_Y + 20), COLOR_WHITE, 1)
    
    # Status text
    status_text = "FACE TRACKING" if tracking.face_detected else "NO FACE"
    status_color = COLOR_GREEN if tracking.face_detected else COLOR_RED
    cv2.putText(overlay, status_text, (FRAME_CENTER_X - 80, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    if face_data:
        # Draw face bounding box
        box_color = COLOR_YELLOW
        cv2.rectangle(overlay, (face_data[0], face_data[1]), 
                      (face_data[2], face_data[3]), 
                      box_color, 2)
        
        # Display servo and offset information
        info_texts = [
            "FACE DETECTED",
            f"X Offset: {face_data['x_offset']:+}%",
            f"Servo Angle: {tracking.current_servo_angle:.1f}°"
        ]
        for i, text in enumerate(info_texts):
            cv2.putText(overlay, text, (w - 200, h - 120 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    # Timestamp
    timestamp = time.strftime("%H:%M:%S %Y-%m-%d", time.localtime())
    cv2.putText(overlay, timestamp, (20, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
    
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame

def process_frame(frame, face_detector):
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(50, 50)
    )
    
    current_time = time.time()
    
    if len(faces) > 0:
        # Sort faces by size and select the largest
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Only process faces of significant size
        if w * h >= 500:
            tracking.face_detected = True
            tracking.last_detection_time = current_time
            
            # Calculate offset and control tracking servo
            x_offset_percent = calculate_face_offset((x, y, x+w, y+h), DISPLAY_WIDTH, DISPLAY_HEIGHT)
            control_tracking_motor(x_offset_percent)
            
            # Prepare face data for drawing
            face_data = {
                'x_offset': x_offset_percent,
                0: x,  # x1
                1: y,  # y1
                2: x+w,  # x2
                3: y+h   # y2
            }
            
            return draw_targeting_interface(frame, face_data)
    
    else:
        # No face detected for more than 0.5 seconds
        if current_time - tracking.last_detection_time > 0.5:
            tracking.face_detected = False
    
    return draw_targeting_interface(frame)

def main():
    face_detector = load_face_detector()
    
    # Initialize PiCamera
    camera = picamera.PiCamera()
    camera.resolution = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    camera.framerate = 15
    
    # Create a video stream
    raw_capture = picamera.array.PiRGBArray(camera, size=(DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    print("Face Tracking System initialized. Press 'q' to exit.")
    
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # Get the numpy array representing the image
            image = frame.array
            
            processed_frame = process_frame(image, face_detector)
            cv2.imshow("Face Tracking System", processed_frame)
            
            # Clear the stream for the next frame
            raw_capture.truncate(0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        print("Cleaning up...")
        camera.close()
        cv2.destroyAllWindows()
        tracking_servo.stop()
        GPIO.cleanup()
        print("Program terminated.")

if __name__ == "__main__":
    main()