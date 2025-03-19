import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
from geopy.distance import geodesic
from PIL import Image

# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Camera GPS coordinates (latitude, longitude)
CAMERA_GPS = (37.7749, -122.4194)

# Known landmarks (GPS coordinates)
LANDMARKS = {
    "Building A": (37.7755, -122.4185),
    "Building B": (37.7740, -122.4200)
}

# Load a smaller, faster YOLO model & enable FP16 precision
model = YOLO("yolov8n.pt").to(device)

# Load MiDaS model in FP16 for speed
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval().half()

# Define optimized image transformations
transform = T.Compose([
    T.Resize((256, 256)),  # Reduced size for faster processing
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(frame, box):
    x1, y1, x2, y2 = box
    drone_crop = frame[y1:y2, x1:x2]

    # Convert NumPy array to PIL image for torchvision
    drone_crop = Image.fromarray(cv2.cvtColor(drone_crop, cv2.COLOR_BGR2RGB))

    # Move input to GPU with FP16
    input_tensor = transform(drone_crop).unsqueeze(0).to(device).half()

    with torch.no_grad():
        depth_map = midas(input_tensor).squeeze().float().cpu().numpy()  # Convert back to FP32

    avg_depth = np.mean(depth_map)

    # Prevent NaN/inf issues
    if np.isnan(avg_depth) or np.isinf(avg_depth):
        avg_depth = 1.0  # Set a safe default depth

    return avg_depth

def triangulate_position(landmark1, landmark2, d1, d2):
    lat1, lon1 = landmark1
    lat2, lon2 = landmark2

    # Convert lat/lon to meters using geodesic distance
    x1, y1 = geodesic((lat1, lon1), (lat1, 0)).meters, geodesic((lat1, lon1), (0, lon1)).meters
    x2, y2 = geodesic((lat2, lon2), (lat2, 0)).meters, geodesic((lat2, lon2), (0, lon2)).meters

    # Ensure valid depth values
    d1, d2 = max(d1, 1.0), max(d2, 1.0)  # Avoid zero distances

    weight1 = d2 / (d1 + d2)
    weight2 = d1 / (d1 + d2)

    x_drone = (weight1 * x1 + weight2 * x2)
    y_drone = (weight1 * y1 + weight2 * y2)

    # Convert back to latitude & longitude
    estimated_lat = lat1 + (x_drone / 111320)
    estimated_lon = lon1 + (y_drone / (111320 * np.cos(np.radians(lat1))))

    return estimated_lat, estimated_lon

# Optimized video capture settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Increase FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB before passing to YOLO (faster processing)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame in batch mode for better GPU efficiency
    results = model(frame_rgb, verbose=False)  # Disable logging for speed

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            depth = estimate_depth(frame, (x1, y1, x2, y2))
            drone_lat, drone_lon = triangulate_position(
                LANDMARKS["Building A"], LANDMARKS["Building B"], depth, depth + 20
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'GPS: {drone_lat:.6f}, {drone_lon:.6f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Drone Detection with GPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
