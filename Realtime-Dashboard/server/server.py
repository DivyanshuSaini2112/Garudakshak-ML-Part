import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
import base64
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
clients = set()
video_clients = set()
data_queue = asyncio.Queue()
video_queue = asyncio.Queue()


@app.websocket("/ws")
async def websocket_client(websocket: WebSocket):
    """Handles WebSocket connections from frontend clients."""
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client connected: {client_id}")
    clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
        clients.remove(websocket)


@app.websocket("/video")
async def websocket_video_client(websocket: WebSocket):
    """Handles WebSocket connections for video feed."""
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Video client connected: {client_id}")
    video_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
    except WebSocketDisconnect:
        logger.info(f"Video client disconnected: {client_id}")
        video_clients.remove(websocket)


@app.websocket("/drone")
async def websocket_drone(websocket: WebSocket):
    """Handles WebSocket connection from the external drone script."""
    await websocket.accept()
    logger.info("Drone script connected")
    try:
        while True:
            data = await websocket.receive_text()  # Receive drone data
            logger.info(
                f"Received drone data: {data[:100]}...")  # Log first 100 chars
            await data_queue.put(data)  # Add it to the queue
    except WebSocketDisconnect:
        logger.info("Drone script disconnected")
    except Exception as e:
        logger.error(f"Error in drone websocket: {str(e)}")


@app.websocket("/video_feed")
async def websocket_video_feed(websocket: WebSocket):
    """Handles WebSocket connection for video feed from drone script."""
    await websocket.accept()
    logger.info("Video feed connected")
    try:
        while True:
            data = await websocket.receive_bytes()  # Receive video frame
            await video_queue.put(data)  # Add it to the video queue
    except WebSocketDisconnect:
        logger.info("Video feed disconnected")
    except Exception as e:
        logger.error(f"Error in video feed websocket: {str(e)}")


async def broadcast_data():
    """Sends queued data to frontend clients."""
    while True:
        if not data_queue.empty():
            data = await data_queue.get()
            logger.info(f"Broadcasting to {len(clients)} clients")
            for client in clients.copy():
                try:
                    await client.send_text(data)
                    logger.info(f"Sent data to client {id(client)}")
                except WebSocketDisconnect:
                    clients.remove(client)
                except Exception as e:
                    logger.error(f"Error sending to client: {str(e)}")
                    clients.remove(client)
        await asyncio.sleep(0.1)


async def broadcast_video():
    """Sends video frames to video clients."""
    while True:
        if not video_queue.empty():
            frame = await video_queue.get()
            for client in video_clients.copy():
                try:
                    await client.send_bytes(frame)
                except WebSocketDisconnect:
                    video_clients.remove(client)
                except Exception as e:
                    logger.error(f"Error sending video to client: {str(e)}")
                    video_clients.remove(client)
        await asyncio.sleep(0.033)  # ~30 FPS


@app.on_event("startup")
async def start_tasks():
    logger.info("Starting broadcast tasks")
    asyncio.create_task(broadcast_data())
    asyncio.create_task(broadcast_video())


@app.get("/")
async def root():
    return {
        "status": "WebSocket server running",
        "clients": len(clients),
        "video_clients": len(video_clients)
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting WebSocket server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
