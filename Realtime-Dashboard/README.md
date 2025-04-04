# 🛰️ Real-time Drone Events Dashboard

This project consists of a **FastAPI WebSocket server** and an **Electron frontend** to display real-time drone event data. Drone event data is streamed in real time from an external Python script (`test5.py`) via WebSockets.

---

## 🗂️ Project Structure

```
.
├── app                     # Electron app
│   ├── assets              # Static assets
│   ├── index.html          # Main HTML entry point
│   ├── main.js             # Electron main process
│   ├── preload.js          # Preload script for secure context bridging
│   ├── renderer.js         # Renderer script that connects to WebSocket
│   ├── style.css           # Styling
│   ├── package.json        # Electron project config
│   └── package-lock.json
├── server                  # FastAPI server
│   └── server.py           # WebSocket server to broadcast drone data
├── README.md               # Project README
```

The `test5.py` drone detection script is located in:
```
../Garudakshak_code/test5.py
```

---

## 🚀 Features

- Real-time drone data streaming via WebSockets
- FastAPI server handles broadcasting to all connected clients
- Electron app renders incoming data in the UI
- Drone simulation script (`test5.py`) pushes data to the system

---

## 📦 Requirements

### Backend (FastAPI server)
- Python 3.11+
- [`fastapi`](https://fastapi.tiangolo.com/)
- [`uvicorn`](https://www.uvicorn.org/)

Install dependencies:

```bash
pip install fastapi uvicorn
```

### Frontend (Electron)
- Node.js 18+
- Electron

Install dependencies:

```bash
cd app
npm install
```

---

## 🧠 How It Works

- **FastAPI server** exposes two WebSocket endpoints:
  - `/ws`: For frontend clients to receive real-time updates
  - `/drone`: For the drone script to send JSON-formatted data
- **Drone script** connects to `/drone` and sends event data
- Server broadcasts that data to all `/ws`-connected clients
- **Electron app** connects to `/ws` and renders the events live

---

## ▶️ Running the Project

### 1. Start the WebSocket Server

```bash
cd server
python server.py
```

This starts the FastAPI server at `http://localhost:8000`.

---

### 2. Run the Electron Frontend

In a separate terminal:

```bash
cd app
npm start
```

This launches the Electron desktop app.

---

### 3. Run the Drone Event Simulator (`test5.py`)

In another terminal, navigate to the drone script directory and run:

```bash
cd ../Garudakshak_code
python test5.py
```

This will start sending live drone events to the server, which are then displayed in the Electron app.

> ✅ Make sure the server is running **before** starting the drone script or frontend.

---

## 🛰️ Manual Testing (Optional)

You can also send test data manually using tools like `websocat`:

```bash
websocat ws://localhost:8000/drone
```

Paste JSON like:

```json
{"timestamp": "2025-04-04T12:00:00", "location": "45.4215,-75.6995", "status": "Flying"}
```

---
