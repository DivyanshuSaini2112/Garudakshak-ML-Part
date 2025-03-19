import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from geopy.distance import geodesic
from PIL import Image
import torch.backends.cudnn as cudnn
from concurrent.futures import ThreadPoolExecutor
import threading

# Tell the GPU to optimize itself for our specific hardware
cudnn.benchmark = True
cudnn.deterministic = False

# Check if we have a GPU and clean up any leftover GPU memory
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Choose between GPU or CPU - GPU is faster if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the camera's location (you should change these to your actual camera location)
CAMERA_GPS = (37.7749, -122.4194)  # Example: San Francisco coordinates

# Set known building locations to help calculate object positions
# Replace these with actual buildings you can see from your camera
LANDMARKS = {
    "Building A": (37.7755, -122.4185),
    "Building B": (37.7740, -122.4200)
}

def classify_flying_object(object_type, box_width, box_height, frame_width, frame_height):
    """
    Determine if a detected airplane should be classified as a drone based on its size and position
    """
    # Calculate relative size of the object compared to frame
    relative_size = (box_width * box_height) / (frame_width * frame_height)
    
    # If it's detected as an airplane but is relatively small (typical for drones)
    if object_type.lower() == "airplane" and relative_size < 0.15:  # Adjust threshold as needed
        return "drone"
    return object_type

try:
    # Load our two main AI models:
    # 1. YOLO: This finds objects in the image
    # 2. MiDaS: This figures out how far away things are
    model = YOLO("yolov8x.pt")
    model.to(device)
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device).eval()
    
    # Keep everything in full precision for better accuracy
    if device.type == 'cuda':
        model.model.float()
        midas.float()
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Missing some required software. Install these with:")
    print("pip install torch torchvision ultralytics opencv-python pillow geopy timm")
    exit(1)

# Set up image processing steps - this prepares images for our AI models
transform = T.Compose([
    T.Resize((256, 256), antialias=True),  # Make all images the same size
    T.ToTensor(),  # Convert image to a format our AI can understand
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standardize the colors
])

# Create a space to store temporary data for each processing thread
thread_local = threading.local()

def get_thread_local_tensor():
    """Create a reusable space in memory to process images"""
    if not hasattr(thread_local, 'tensor'):
        thread_local.tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
    return thread_local.tensor

@torch.inference_mode()
def estimate_depth(frame, box):
    """Figure out how far away an object is in the image"""
    try:
        # Get the part of the image containing our object
        x1, y1, x2, y2 = box
        object_image = frame[y1:y2, x1:x2]
        
        # Skip if the object is too small to measure accurately
        if object_image.shape[0] < 10 or object_image.shape[1] < 10:
            return 1.0

        # Convert the image to the right format for our depth-sensing AI
        object_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
        
        # Process the image and get the depth
        input_tensor = get_thread_local_tensor()
        with torch.no_grad():
            input_data = transform(object_image).to(device, dtype=torch.float32, non_blocking=True)
            input_tensor[0] = input_data
            
            depth_map = midas(input_tensor)
            average_depth = torch.mean(depth_map).item()
        
        return max(average_depth, 1.0)
    except Exception as e:
        print(f"Problem measuring depth: {str(e)}")
        return 1.0

def triangulate_position(landmark1, landmark2, d1, d2):
    """Calculate the GPS position of an object using nearby landmarks"""
    try:
        # Get the coordinates of our reference buildings
        lat1, lon1 = landmark1
        lat2, lon2 = landmark2
        
        # Convert latitude to radians for math calculations
        lat1_rad = np.radians(lat1)
        
        # Calculate positions in meters instead of coordinates
        x1 = np.cos(lat1_rad) * 111320 * (lon1)
        y1 = lat1 * 111320
        x2 = np.cos(lat1_rad) * 111320 * (lon2)
        y2 = lat2 * 111320
        
        # Use weighted averages based on distance to estimate position
        total_distance = d1 + d2
        weight1 = d2 / total_distance
        weight2 = d1 / total_distance
        
        x_obj = (weight1 * x1 + weight2 * x2)
        y_obj = (weight1 * y1 + weight2 * y2)
        
        # Convert back to GPS coordinates
        return (y_obj / 111320, x_obj / (111320 * np.cos(lat1_rad)))
    except Exception as e:
        print(f"Problem calculating position: {str(e)}")
        return CAMERA_GPS

try:
    # Start the camera
    cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam, 1 or 2 for external cameras
    if not cap.isOpened():
        raise Exception("Couldn't access the camera")
        
    # Set up the camera for good performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width of the video
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height of the video
    cap.set(cv2.CAP_PROP_FPS, 30)            # Frames per second
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize delay

    # Set up multiple processing threads to speed things up
    executor = ThreadPoolExecutor(max_workers=4)

    # Create a reusable space in memory for processing
    frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)

    print("Camera is ready! Press 'q' to stop the program.")
    
    # Set up performance tracking
    frame_count = 0
    start_time = cv2.getTickCount()

    # Main loop - this runs continuously until you press 'q'
    while True:
        # Get a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Problem getting image from camera")
            break
            
        # Copy the frame to our processing buffer
        np.copyto(frame_buffer, frame)
        
        # Look for objects in the frame
        results = model(frame_buffer, conf=0.4, iou=0.45, max_det=10)
        
        frame_height, frame_width = frame.shape[:2]
        futures = []
        
        # Process each object we found
        for result in results:
            for box in result.boxes:
                # Get the location and type of each object
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                object_type = model.names[int(box.cls[0])]
                
                # Calculate box dimensions
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Skip objects that are too small
                if box_width * box_height < 100:
                    continue
                
                # Refine classification for flying objects
                object_type = classify_flying_object(object_type, box_width, box_height, 
                                                  frame_width, frame_height)
                    
                # Start calculating the depth for this object
                future = executor.submit(estimate_depth, frame_buffer, (x1, y1, x2, y2))
                futures.append((future, (x1, y1, x2, y2), object_type, confidence))
        
        # Draw boxes and information for each object
        for future, box, object_type, confidence in futures:
            depth = future.result()
            x1, y1, x2, y2 = box
            
            # Calculate the GPS position of the object
            obj_lat, obj_lon = triangulate_position(
                LANDMARKS["Building A"], 
                LANDMARKS["Building B"], 
                depth, 
                depth + 20
            )
            
            # Red boxes for drones and airplanes, green for everything else
            color = (0, 0, 255) if object_type.lower() in ["drone", "airplane"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add labels with object type and GPS position
            texts = [
                (f"{object_type} ({confidence:.2f})", (x1, y1 - 20)),
                (f"GPS: {obj_lat:.6f}, {obj_lon:.6f}", (x1, y1 - 5))
            ]
            for text, pos in texts:
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate and show frames per second (FPS)
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            fps = frame_count * cv2.getTickFrequency() / (current_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the processed image
        cv2.imshow("Object Detection with GPS", frame)
        
        # Check if 'q' was pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Clean up everything when we're done
    print("Cleaning up...")
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    if 'executor' in locals():
        executor.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()