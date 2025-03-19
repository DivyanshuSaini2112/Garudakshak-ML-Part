import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\Visual Code\Garudakshak_code\yolov8x.pt")

# Open the video source
video_path = "D:\Visual Code\Garudakshak_code\cam1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    
    # Fix flipped video
    frame = cv2.flip(frame, 0)  # Flip back to correct orientation
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Run inference with a lower confidence threshold
    results = model(frame, conf=0.3)
    
    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write frame to output file
    out.write(frame)
    
    # Display the frame
    cv2.imshow("YOLO Detection", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
