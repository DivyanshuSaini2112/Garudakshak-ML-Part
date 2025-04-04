import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg


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
        # Added for dashboard
        self.gps_history = []
        self.distance_history = []
        self.speed_history = []
        self.lock_status_history = []
        self.time_history = []

tracking = TrackingState()

def calculate_centering_data(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    x_offset = object_center_x - FRAME_CENTER_X
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
        print("Required packages: torch torchvision opencv-python pillow geopy matplotlib")
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

def generate_dashboard(tracking_data):
    """
    Generate a professional visualization dashboard for drone tracking data with Garudakshak branding.
    
    Parameters:
    tracking_data (TrackingState): Object containing the tracking history
    
    Returns:
    numpy.ndarray: Image of the dashboard in BGR format for OpenCV display
    """
    # Garudakshak theme
    GARUDAKSHAK_BLUE = '#00285e'
    GARUDAKSHAK_ORANGE = '#ff9966'
    GARUDAKSHAK_WHITE = '#ffffff'
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': GARUDAKSHAK_BLUE,
        'axes.facecolor': GARUDAKSHAK_BLUE,
        'axes.edgecolor': GARUDAKSHAK_WHITE,
        'axes.labelcolor': GARUDAKSHAK_WHITE,
        'xtick.color': GARUDAKSHAK_WHITE,
        'ytick.color': GARUDAKSHAK_WHITE,
        'text.color': GARUDAKSHAK_WHITE,
        'axes.grid': True,
        'grid.color': GARUDAKSHAK_WHITE,
        'grid.alpha': 0.2,
        'lines.linewidth': 2.5,
        'font.family': 'sans-serif',
        'font.weight': 'bold'
    })
    
    fig = plt.figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.5], width_ratios=[2, 1])

    time_values = [datetime.datetime.fromtimestamp(t) for t in tracking_data.time_history]
    
    # Ensure all histories are the same length
    min_length = min(len(tracking_data.time_history), len(tracking_data.gps_history),
                     len(tracking_data.distance_history), len(tracking_data.speed_history),
                     len(tracking_data.lock_status_history))
    time_values = time_values[:min_length]
    gps_history = tracking_data.gps_history[:min_length]
    distance_history = tracking_data.distance_history[:min_length]
    speed_history = tracking_data.speed_history[:min_length]
    lock_status_history = tracking_data.lock_status_history[:min_length]
    
    # 1. GPS Tracking Path (Top Left)
    ax_gps = fig.add_subplot(gs[0, 0])
    if gps_history:
        def gps_to_relative(lat, lon):
            camera_lat, camera_lon = CAMERA_GPS
            lat_diff = (lat - camera_lat) * 111320  # meters North
            lon_diff = (lon - camera_lon) * 111320 * np.cos(np.radians(camera_lat))  # meters East
            return lon_diff, lat_diff  # East, North
        
        rel_x, rel_y = zip(*[gps_to_relative(lat, lon) for lat, lon in gps_history])
        ax_gps.plot(rel_x, rel_y, color='#0088ff', label='Target Path')
        ax_gps.plot(0, 0, marker='o', markersize=12, color='#ff0000', label='Camera')
        
        for zone_name, (zone_lat, zone_lon) in LANDMARKS.items():
            zx, zy = gps_to_relative(zone_lat, zone_lon)
            ax_gps.plot(zx, zy, marker='o', markersize=10, color=GARUDAKSHAK_ORANGE, 
                        label=zone_name if zone_name == "Zone A" else "")
        
        if tracking_data.target_gps:
            tx, ty = gps_to_relative(tracking_data.target_gps[0], tracking_data.target_gps[1])
            ax_gps.plot(tx, ty, marker='*', markersize=15, color='#ff0000', label='Current Target')
        
        x_margin = max(5, (max(rel_x) - min(rel_x)) * 0.1) if rel_x else 10
        y_margin = max(5, (max(rel_y) - min(rel_y)) * 0.1) if rel_y else 10
        ax_gps.set_xlim(min(rel_x) - x_margin, max(rel_x) + x_margin) if rel_x else (-10, 10)
        ax_gps.set_ylim(min(rel_y) - y_margin, max(rel_y) + y_margin) if rel_y else (-10, 10)
    
    ax_gps.set_title('GPS Tracking Path', fontsize=14, fontweight='bold', pad=10)
    ax_gps.set_xlabel('Distance East (m)', fontsize=11)
    ax_gps.set_ylabel('Distance North (m)', fontsize=11)
    ax_gps.grid(True, alpha=0.2)
    ax_gps.legend(loc='upper right', fontsize=10, framealpha=0.7)

    # 2. Target Distance (Top Right)
    ax_dist = fig.add_subplot(gs[0, 1])
    if distance_history:
        ax_dist.plot(time_values, distance_history, color='#00cc44', label='Distance (m)')
    ax_dist.set_title('Target Distance', fontsize=14, fontweight='bold', pad=10)
    ax_dist.set_ylabel('Distance (m)', fontsize=11)
    ax_dist.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_dist.grid(True, alpha=0.2)
    ax_dist.legend(loc='upper right', fontsize=10, framealpha=0.7)
    plt.setp(ax_dist.get_xticklabels(), rotation=45, fontsize=9)

    # 3. Target Velocity (Bottom Left)
    ax_vel = fig.add_subplot(gs[1, 0])
    if speed_history:
        # Simple moving average for smoothing
        smoothed_speed = np.convolve(speed_history, np.ones(5)/5, mode='valid')
        smoothed_time = time_values[len(time_values) - len(smoothed_speed):]
        ax_vel.plot(smoothed_time, smoothed_speed, color='#0088ff', label='Velocity (m/s)')
    ax_vel.set_title('Target Velocity', fontsize=14, fontweight='bold', pad=10)
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=11)
    ax_vel.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_vel.grid(True, alpha=0.2)
    ax_vel.legend(loc='upper right', fontsize=10, framealpha=0.7)
    plt.setp(ax_vel.get_xticklabels(), rotation=45, fontsize=9)

    # 4. Target Lock Status (Bottom Right)
    ax_lock = fig.add_subplot(gs[1, 1])
    if lock_status_history:
        ax_lock.step(time_values, lock_status_history, where='post', color=GARUDAKSHAK_ORANGE)
        ax_lock.fill_between(time_values, 0, lock_status_history, step='post', 
                            alpha=0.4, color=GARUDAKSHAK_ORANGE)
    ax_lock.set_title('Target Lock Status', fontsize=14, fontweight='bold', pad=10)
    ax_lock.set_ylabel('Status', fontsize=11)
    ax_lock.set_ylim(-0.1, 1.1)
    ax_lock.set_yticks([0, 1])
    ax_lock.set_yticklabels(['Unlocked', 'Locked'], fontsize=10)
    ax_lock.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_lock.grid(True, alpha=0.2)
    plt.setp(ax_lock.get_xticklabels(), rotation=45, fontsize=9)

    # Summary Table (outside the plot)
    summary_data = {
        'Metrics': ['Distance (m)', 'Velocity (m/s)', 'Lock Status'],
        'Value': [
            f"{tracking_data.target_distance:.1f}" if tracking_data.target_distance else 'N/A',
            f"{speed_history[-1]:.1f}" if speed_history else 'N/A',
            'Locked' if tracking_data.locked else 'Unlocked'
        ]
    }
    table = plt.table(cellText=[summary_data['Value']], colLabels=summary_data['Metrics'],
                     loc='bottom', bbox=[0.75, -0.25, 0.2, 0.15], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='#333333')
            cell.set_facecolor('#e0e0e0')
        else:
            cell.set_facecolor(GARUDAKSHAK_ORANGE if j == 2 else '#001f46')
            cell.set_text_props(color=GARUDAKSHAK_WHITE)
    
    # Branding and metadata
    fig.text(0.5, 0.96, 'GARUDAKSHAK', fontsize=22, fontweight='bold', 
             color=GARUDAKSHAK_WHITE, ha='center')
    fig.text(0.5, 0.925, 'Anti-Drone Detection System', fontsize=14, 
             color=GARUDAKSHAK_ORANGE, ha='center', style='italic')
    if tracking_data.target_gps:
        gps_text = f"Current GPS: {tracking_data.target_gps[0]:.6f}, {tracking_data.target_gps[1]:.6f}"
        fig.text(0.01, 0.975, gps_text, fontsize=9, color=GARUDAKSHAK_WHITE,
                 bbox=dict(fc='#001f46', ec=GARUDAKSHAK_ORANGE, lw=1))
    timestamp = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    fig.text(0.5, 0.01, f'Last Updated: {timestamp}', ha='center', fontsize=9, 
             color='#aaaaaa', style='italic')

    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.9])
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img

# Update process_frame to ensure consistent history lengths
def process_frame(frame, face_detector, midas_model, executor):
    # ... (previous code remains largely unchanged until history updates)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    num_targets = len(faces)  # Define num_targets as the number of detected faces
    if num_targets > 0:
        # ... (detection and calculation logic)
        
        # Update all histories together
        current_data = {
            'gps': tracking.target_gps,
            'distance': tracking.target_distance,
            'speed': 0,  # Default value for speed when not defined
            'lock_status': 1 if tracking.locked else 0,
            'time': time.time()  # Define current_time using time.time()
        }
        tracking.gps_history.append(current_data['gps'])
        tracking.distance_history.append(current_data['distance'])
        tracking.speed_history.append(current_data['speed'])
        tracking.lock_status_history.append(current_data['lock_status'])
        tracking.time_history.append(current_data['time'])
    else:
        # Update with no-target data
        current_data = {
            'gps': tracking.target_gps if tracking.target_gps else tracking.gps_history[-1] if tracking.gps_history else CAMERA_GPS,
            'distance': tracking.target_distance,
            'speed': 0,
            'lock_status': 0,
            'time': time.time()  # Define current_time using time.time()
        }
        tracking.gps_history.append(current_data['gps'])
        tracking.distance_history.append(current_data['distance'])
        tracking.speed_history.append(current_data['speed'])
        tracking.lock_status_history.append(current_data['lock_status'])
        tracking.time_history.append(current_data['time'])
    
    # ... (rest of the function remains the same)

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
    
    if num_targets > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        box = (x, y, x+w, y+h)
        
        if w * h >= 1000:
            target_data = calculate_centering_data(box, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            tracking.last_detection_time = current_time
            tracking.target_data = target_data  # Store for dashboard
            
            future = executor.submit(estimate_depth, frame, box, midas_model)
            try:
                tracking.target_distance = future.result(timeout=0.1)
            except:
                tracking.target_distance = 1.0
            
            tracking.target_gps = triangulate_position(tracking.target_distance, target_data)
            
            if tracking.last_position and tracking.target_gps:
                elapsed_time = current_time - tracking.last_time
                dist = geodesic(tracking.last_position, tracking.target_gps).meters
                speed = dist / elapsed_time if elapsed_time > 0 else 0
            else:
                speed = 0
            tracking.last_position = tracking.target_gps
            
            # Update tracking history
            tracking.target_history.append(target_data)
            tracking.gps_history.append(tracking.target_gps)
            tracking.distance_history.append(tracking.target_distance)
            tracking.speed_history.append(speed)
            tracking.lock_status_history.append(1 if tracking.locked else 0)
            tracking.time_history.append(current_time)
            
            if target_data["is_centered"]:
                tracking.lock_counter = min(tracking.lock_counter + 1, TARGET_LOCKED_THRESHOLD)
                if tracking.lock_counter == TARGET_LOCKED_THRESHOLD and not tracking.locked:
                    print("TARGET LOCKED!")
                    tracking.locked = True
            else:
                tracking.lock_counter = max(tracking.lock_counter - 1, 0)
                if tracking.lock_counter == 0 and tracking.locked:
                    print("Lock lost")
                    tracking.locked = False
    else:
        speed = 0
        if current_time - tracking.last_detection_time > 0.5:
            if tracking.locked:
                print("No target detected. Lock released.")
            tracking.locked = False
            tracking.lock_counter = 0
            tracking.target_history = []
        
        # Still record data even when no target
        tracking.distance_history.append(tracking.target_distance)
        tracking.speed_history.append(speed)
        tracking.lock_status_history.append(0)
        tracking.time_history.append(current_time)
        if tracking.target_gps:
            tracking.gps_history.append(tracking.target_gps)
    
    processed_frame = draw_targeting_interface(frame, target_data)
    postprocess_time = (time.time() - postprocess_start) * 1000
    total_time = (time.time() - start_time) * 1000
    
    target_text = f"{num_targets} target{'s' if num_targets != 1 else ''}"
    gps_text = f"GPS: {tracking.target_gps[0]:.6f}, {tracking.target_gps[1]:.6f}" if tracking.target_gps else "GPS: None"
    
    if target_data:
        x_instruction = (f"Move {'left' if target_data['x_offset_percent'] < 0 else 'right'} "
                        f"{abs(target_data['x_offset_percent'])}%")
        y_instruction = (f"Move {'down' if target_data['y_offset_percent'] < 0 else 'up'} "
                        f"{abs(target_data['y_offset_percent'])}%")
        movement_text = f"{x_instruction}, {y_instruction}"
    else:
        movement_text = "No adjustment needed"
    
    print(f"0: {DISPLAY_HEIGHT}x{DISPLAY_WIDTH} {target_text}, {total_time:.1f}ms")
    print(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, "
          f"{postprocess_time:.1f}ms postprocess per image at shape (1, 3, {DISPLAY_HEIGHT}, {DISPLAY_WIDTH})")
    print(f"Position: {gps_text}, Adjustment: {movement_text}, Distance: {tracking.target_distance:.1f}m\n")
    
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
    
    print("Anti-Drone Detection System initialized. Press 'q' to exit and view dashboard.")
    
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
            print("Cleaning up camera feed...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Generate and display dashboard
            if tracking.time_history:  # Only show dashboard if there's data
                print("Generating tracking dashboard...")
                dashboard_img = generate_dashboard(tracking)
                cv2.imshow("Tracking Dashboard", dashboard_img)
                print("Dashboard displayed. Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Program terminated.")

if __name__ == "__main__":
    main()