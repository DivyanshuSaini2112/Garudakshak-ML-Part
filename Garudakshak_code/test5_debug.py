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
import websockets
import asyncio
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimize for performance
cudnn.benchmark = True
cudnn.deterministic = False
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# GPS Settings
CAMERA_GPS = (37.7749, -122.4194)
LANDMARKS = {"Zone A": (37.7755, -122.4185), "Zone B": (37.7740, -122.4200)}

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
        self.processed_frame = None
        # Added for dashboard
        self.gps_history = []
        self.distance_history = []
        self.speed_history = []
        self.lock_status_history = []
        self.time_history = []


tracking = TrackingState()


async def send_video_feed():
    """Send video frames to the WebSocket server."""
    uri = "ws://localhost:8000/video_feed"
    logger.info("Attempting to connect to video feed WebSocket...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to video feed WebSocket")
            while True:
                if tracking.processed_frame is not None:
                    # Convert frame to JPEG
                    _, buffer = cv2.imencode('.jpg', tracking.processed_frame,
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])
                    # Send frame as bytes
                    await websocket.send(buffer.tobytes())
                await asyncio.sleep(0.033)  # ~30 FPS
    except Exception as e:
        logger.error(f"Video feed error: {str(e)}")
        # Try to reconnect after 5 seconds
        await asyncio.sleep(5)
        asyncio.create_task(send_video_feed())


async def send_drone_data():
    """Send drone data to the WebSocket server."""
    uri = "ws://localhost:8000/drone"
    logger.info("Attempting to connect to drone data WebSocket...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to drone data WebSocket")
            while True:
                if tracking.target_gps:
                    data = {
                        "gps":
                        tracking.target_gps,
                        "distance":
                        tracking.target_distance,
                        "speed":
                        tracking.speed_history[-1]
                        if tracking.speed_history else 0,
                        "locked":
                        tracking.locked,
                        "timestamp":
                        datetime.datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(data))
                await asyncio.sleep(0.1)  # 10 Hz update rate
    except Exception as e:
        logger.error(f"Drone data error: {str(e)}")
        # Try to reconnect after 5 seconds
        await asyncio.sleep(5)
        asyncio.create_task(send_drone_data())


def load_models():
    logger.info("Loading models...")
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device).eval()
        if device.type == 'cuda':
            midas.float()
        logger.info("Models loaded successfully")
        return face_cascade, midas
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


async def main_async():
    logger.info("Starting main async function")
    face_detector, midas_model = load_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open video capture")
        return

    logger.info("Video capture opened successfully")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info(
        "Anti-Drone Detection System initialized. Press 'q' to exit and view dashboard."
    )

    # Start WebSocket tasks
    logger.info("Starting WebSocket tasks")
    asyncio.create_task(send_video_feed())
    asyncio.create_task(send_drone_data())

    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: Failed to capture frame")
                    break

                # Simple frame processing for testing
                processed_frame = cv2.resize(frame,
                                             (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                tracking.processed_frame = processed_frame

                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - start_time)
                    start_time = time.time()
                    logger.info(f"Current FPS: {fps:.1f}")
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                                (DISPLAY_WIDTH - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

                cv2.imshow("Anti-Drone Detection System", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received")
                    break

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            logger.info("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Program terminated.")


def main():
    logger.info("Starting program")
    try:
        asyncio.run(main_async())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
