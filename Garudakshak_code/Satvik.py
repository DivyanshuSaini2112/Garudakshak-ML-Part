import cv2
import numpy as np
import torch
import torchvision.transforms as T
from timm.models import create_model
from ultralytics import YOLO
from geopy.distance import geodesic

# Camera GPS coordinates (latitude, longitude)
CAMERA_GPS = (37.7749, -122.4194)

# Known landmarks (GPS coordinates)
LANDMARKS = {
    "Building A": (37.7755, -122.4185),
    "Building B": (37.7740, -122.4200)
}

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load MiDaS model
midas = create_model("DPT_Large", pretrained=True).eval()
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(frame, box):
    x1, y1, x2, y2 = box
    drone_crop = frame[y1:y2, x1:x2]
    input_tensor = transform(drone_crop).unsqueeze(0)
    with torch.no_grad():
        depth_map = midas(input_tensor).squeeze().numpy()
    return np.mean(depth_map)

def triangulate_position(landmark1, landmark2, d1, d2):
    lat1, lon1 = landmark1
    lat2, lon2 = landmark2
    x1, y1 = geodesic((lat1, lon1), (lat1, 0)).meters, geodesic((lat1, lon1), (0, lon1)).meters
    x2, y2 = geodesic((lat2, lon2), (lat2, 0)).meters, geodesic((lat2, lon2), (0, lon2)).meters
    A, B = 2 * (x2 - x1), 2 * (y2 - y1)
    C = d1*2 - d22 - x12 + x22 - y12 + y2*2
    x_drone = x1 + (C * (x2 - x1)) / (A + B)
    y_drone = y1 + (C * (y2 - y1)) / (A + B)
    estimated_lat = lat1 + (x_drone / 111320)
    estimated_lon = lon1 + (y_drone / (111320 * np.cos(np.radians(lat1))))
    return estimated_lat, estimated_lon

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            depth = estimate_depth(frame, (x1, y1, x2, y2))
            drone_lat, drone_lon = triangulate_position(LANDMARKS["Building A"], LANDMARKS["Building B"], depth, depth + 20)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'GPS: {drone_lat:.6f}, {drone_lon:.6f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Drone Detection with GPS", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()