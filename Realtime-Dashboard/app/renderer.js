// renderer.js - Frontend code for the Electron app

// DOM Elements
const connectionIndicator = document.getElementById('connection-indicator');
const connectionText = document.getElementById('connection-text');
const dataToggle = document.getElementById('data-toggle');
const alertBar = document.getElementById('alert-bar');
const alertMessage = document.getElementById('alert-message');
const dismissAlert = document.getElementById('dismiss-alert');
const detectionMap = document.getElementById('detection-map');
const droneMarker = document.getElementById('drone-marker');
const homeMarker = document.getElementById('home-marker');
const coordinatesDisplay = document.getElementById('coordinates');
const distanceValue = document.getElementById('distance-value');
const speedValue = document.getElementById('speed-value');
const timeValue = document.getElementById('time-value');
const detectionStatus = document.getElementById('detection-status');
const statsInfo = document.getElementById('stats-info');
const movementStatus = document.getElementById('movement-status');
const preprocessTime = document.getElementById('preprocess-time');
const inferenceTime = document.getElementById('inference-time');
const postprocessTime = document.getElementById('postprocess-time');
const lastUpdated = document.getElementById('last-updated');
const eventLog = document.getElementById('event-log');
const clearLogBtn = document.getElementById('clear-log');
const settingsBtn = document.getElementById('settings-btn');
const mapContainer = document.getElementById('map-container');

// WebSocket connection
let socket = null;
let isConnected = false;
let isMockDataEnabled = false;
let mockDataInterval = null;
const SERVER_URL = 'ws://localhost:8000/ws';

// Map center coordinates (home position)
let mapCenterX = mapContainer.offsetWidth / 2;
let mapCenterY = mapContainer.offsetHeight / 2;

// Chart initialization with responsive options
const historyCtx = document.getElementById('history-chart').getContext('2d');
const historyChart = new Chart(historyCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Distance (m)',
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        data: [],
        tension: 0.3,
        yAxisID: 'y'
      },
      {
        label: 'Speed (km/h)',
        borderColor: '#F87171',
        backgroundColor: 'rgba(248, 113, 113, 0.1)',
        data: [],
        tension: 0.3,
        yAxisID: 'y1'
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          padding: 10,
          color: '#E5E7EB'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.8)',
        titleColor: '#F3F4F6',
        bodyColor: '#E5E7EB',
        borderColor: '#4B5563',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(55, 65, 81, 0.3)'
        },
        ticks: {
          color: '#9CA3AF',
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 8
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Distance (m)',
          color: '#9CA3AF',
          font: {
            size: 10
          }
        },
        grid: {
          color: 'rgba(55, 65, 81, 0.3)'
        },
        ticks: {
          color: '#9CA3AF'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Speed (km/h)',
          color: '#9CA3AF',
          font: {
            size: 10
          }
        },
        grid: {
          drawOnChartArea: false,
          color: 'rgba(55, 65, 81, 0.3)'
        },
        ticks: {
          color: '#9CA3AF'
        }
      }
    }
  }
});

// Position the home marker initially
function setMapHomePosition() {
  // Update center coords based on current map dimensions
  mapCenterX = mapContainer.offsetWidth / 2;
  mapCenterY = mapContainer.offsetHeight / 2;
  
  // Position home marker at center
  homeMarker.style.left = `${mapCenterX}px`;
  homeMarker.style.top = `${mapCenterY}px`;
  
  // Update ring positions if they exist
  const rings = document.querySelectorAll('.distance-ring');
  rings.forEach(ring => {
    ring.style.top = '50%';
    ring.style.left = '50%';
  });
}

// Initialize the app
function init() {
  addLogEntry('System initialized', 'info');
  setUpEventListeners();
  
  // Set initial toggle state
  dataToggle.checked = false;
  toggleDataSource();
  
  // Initialize map position
  setMapHomePosition();
  
  // Make sure the map is properly sized
  window.dispatchEvent(new Event('resize'));
}

// Set up event listeners
function setUpEventListeners() {
  // Toggle data source
  dataToggle.addEventListener('change', toggleDataSource);
  
  // Dismiss alert
  dismissAlert.addEventListener('click', () => {
    alertBar.classList.add('hidden');
  });
  
  // Clear log
  clearLogBtn.addEventListener('click', () => {
    eventLog.innerHTML = '';
    addLogEntry('Log cleared', 'info');
  });
  
  // Settings button
  settingsBtn.addEventListener('click', () => {
    addLogEntry('Settings clicked - functionality not implemented', 'info');
  });
  
  // Handle window resize
  window.addEventListener('resize', () => {
    // Properly resize the map
    setMapHomePosition();
    
    // Force chart to properly resize
    historyChart.resize();
    
    // If we have drone position data, update it
    if (!droneMarker.classList.contains('hidden')) {
      updateDronePosition(lastDroneData);
    }
  });
}

// Store last drone data
let lastDroneData = null;

// Toggle between live and mock data
function toggleDataSource() {
  isMockDataEnabled = !dataToggle.checked;
  
  if (isMockDataEnabled) {
    // Disconnect WebSocket if connected
    if (socket) {
      socket.close();
      socket = null;
    }
    
    // Start mock data generation
    startMockData();
    updateConnectionStatus(false, true);
    addLogEntry('Switched to mock data mode', 'info');
  } else {
    // Stop mock data generation
    if (mockDataInterval) {
      clearInterval(mockDataInterval);
      mockDataInterval = null;
    }
    
    // Connect to WebSocket
    connectWebSocket();
    addLogEntry('Connecting to live data...', 'info');
  }
}

// Connect to WebSocket server
function connectWebSocket() {
  if (socket) {
    socket.close();
  }
  
  updateConnectionStatus(false);
  
  socket = new WebSocket(SERVER_URL);
  
  socket.onopen = () => {
    updateConnectionStatus(true);
    addLogEntry('Connected to server', 'success');
  };
  
  socket.onmessage = (event) => {
    processDroneData(JSON.parse(event.data));
  };
  
  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    addLogEntry('Connection error', 'alert');
    updateConnectionStatus(false);
  };
  
  socket.onclose = () => {
    console.log('WebSocket connection closed');
    addLogEntry('Connection closed', 'info');
    updateConnectionStatus(false);
    
    // Try to reconnect after 5 seconds if not in mock mode
    if (!isMockDataEnabled) {
      setTimeout(connectWebSocket, 5000);
    }
  };
}

// Update connection status indicator
function updateConnectionStatus(connected, isMock = false) {
  isConnected = connected;
  
  if (isMock) {
    connectionIndicator.className = 'h-3 w-3 rounded-full bg-yellow-500 mr-2';
    connectionText.textContent = 'Mock Data';
    connectionText.className = 'text-yellow-500';
    return;
  }
  
  if (connected) {
    connectionIndicator.className = 'h-3 w-3 rounded-full bg-green-500 mr-2';
    connectionText.textContent = 'Connected';
    connectionText.className = 'text-green-500';
  } else {
    connectionIndicator.className = 'h-3 w-3 rounded-full bg-red-500 mr-2';
    connectionText.textContent = 'Disconnected';
    connectionText.className = 'text-red-500';
  }
}

// Start generating mock data
function startMockData() {
  if (mockDataInterval) {
    clearInterval(mockDataInterval);
  }
  
  let angle = 0;
  let distance = 50;
  let increasing = true;
  
  mockDataInterval = setInterval(() => {
    // Create mock drone data
    angle = (angle + 5) % 360;
    
    if (increasing) {
      distance += Math.random() * 5;
      if (distance > 140) increasing = false;
    } else {
      distance -= Math.random() * 5;
      if (distance < 50) increasing = true;
    }
    
    const speed = 10 + Math.random() * 40;
    const timestamp = new Date().toISOString();
    
    // Generate random GPS coordinates near a base position
    const baseLatitude = 37.7749;
    const baseLongitude = -122.4194;
    const randomLat = baseLatitude + (Math.random() - 0.5) * 0.02;
    const randomLong = baseLongitude + (Math.random() - 0.5) * 0.02;
    
    const mockData = {
      stats: Math.random() > 0.5 ? "Drone detected with high confidence" : "Potential drone signature detected",
      time: (Math.random() * 120).toFixed(1),
      speed: speed,
      timestamp: timestamp,
      position: `GPS: ${randomLat.toFixed(6)}° N, ${randomLong.toFixed(6)}° E`,
      adjustment: ['Approaching', 'Moving away', 'Circling', 'Hovering'][Math.floor(Math.random() * 4)],
      distance: distance,
      processing_times: {
        preprocess: Math.random() * 10,
        inference: Math.random() * 50 + 30,
        postprocess: Math.random() * 15
      },
      // Adding angle for visualization
      angle: angle
    };
    
    processDroneData(mockData);
  }, 1000);
}

// Process incoming drone data
function processDroneData(data) {
  // Store the data for potential reuse
  lastDroneData = data;
  
  // Update last updated timestamp
  lastUpdated.textContent = `Last updated: ${moment().format('HH:mm:ss')}`;
  
  // Update UI elements with drone data
  updateDronePosition(data);
  updateMetrics(data);
  updateStats(data);
  updateChart(data);
  
  // Log the event
  addLogEntry(`Data received: ${data.stats}`, 'info');
  
  // Show alert if drone is too close
  if (data.distance < 70) {
    showAlert(`Alert: Drone detected ${data.distance.toFixed(0)}m away! ${data.adjustment}`);
  }
}

// Update drone position on the map
function updateDronePosition(data) {
  if (!data) return;
  
  // Make sure the drone marker is visible
  droneMarker.classList.remove('hidden');
  
  // Scale factor to convert distance to pixels (150m = map radius)
  const maxMapRadius = Math.min(mapContainer.offsetWidth, mapContainer.offsetHeight) / 2;
  const scaleFactor = maxMapRadius / 150;
  
  // Calculate position based on distance and angle
  // If angle is not provided (live data), use a calculated one
  const angle = data.angle !== undefined ? 
    data.angle * (Math.PI / 180) : 
    (Date.now() / 10000) % (2 * Math.PI);
  
  const scaledDistance = data.distance * scaleFactor;
  const x = mapCenterX + scaledDistance * Math.cos(angle);
  const y = mapCenterY + scaledDistance * Math.sin(angle);
  
  // Update drone marker position
  droneMarker.style.left = `${x}px`;
  droneMarker.style.top = `${y}px`;
  
  // Update coordinates display
  coordinatesDisplay.textContent = data.position;
}

// Update metrics display
function updateMetrics(data) {
  distanceValue.textContent = `${data.distance.toFixed(1)} m`;
  speedValue.textContent = `${data.speed.toFixed(1)} km/h`;
  timeValue.textContent = `${data.time} s`;
  
  // Update processing times
  preprocessTime.textContent = `${data.processing_times.preprocess.toFixed(1)}`;
  inferenceTime.textContent = `${data.processing_times.inference.toFixed(1)}`;
  postprocessTime.textContent = `${data.processing_times.postprocess.toFixed(1)}`;
}

// Update stats and status information
function updateStats(data) {
  // Status changes based on distance
  if (data.distance < 70) {
    detectionStatus.textContent = 'DRONE DETECTED';
    detectionStatus.className = 'text-lg font-bold text-red-500';
  } else if (data.distance < 120) {
    detectionStatus.textContent = 'Drone Detected';
    detectionStatus.className = 'text-lg font-bold text-yellow-500';
  } else {
    detectionStatus.textContent = 'Monitoring';
    detectionStatus.className = 'text-lg font-bold text-green-500';
  }
  
  statsInfo.textContent = data.stats;
  movementStatus.textContent = data.adjustment;
  
  // Add some visual flair to movement status
  switch (data.adjustment) {
    case 'Approaching':
      movementStatus.className = 'text-lg font-bold text-red-400';
      break;
    case 'Moving away':
      movementStatus.className = 'text-lg font-bold text-green-400';
      break;
    case 'Circling':
      movementStatus.className = 'text-lg font-bold text-yellow-400';
      break;
    case 'Hovering':
      movementStatus.className = 'text-lg font-bold text-blue-400';
      break;
    default:
      movementStatus.className = 'text-lg font-bold';
  }
}

// Update history chart
function updateChart(data) {
  const timestamp = moment().format('HH:mm:ss');
  
  // Add data to chart
  historyChart.data.labels.push(timestamp);
  historyChart.data.datasets[0].data.push(data.distance);
  historyChart.data.datasets[1].data.push(data.speed);
  
  // Keep only last 15 data points to prevent overcrowding
  if (historyChart.data.labels.length > 15) {
    historyChart.data.labels.shift();
    historyChart.data.datasets.forEach(dataset => dataset.data.shift());
  }
  
  historyChart.update();
}

// Show alert message
function showAlert(message) {
  alertMessage.textContent = message;
  alertBar.classList.remove('hidden');
  addLogEntry(message, 'alert');
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    alertBar.classList.add('hidden');
  }, 5000);
}

// Add entry to event log
function addLogEntry(message, type = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type} fade-in`;
  entry.innerHTML = `<span class="text-gray-400">[${moment().format('HH:mm:ss')}]</span> ${message}`;
  eventLog.prepend(entry);
  
  // Keep log size manageable
  if (eventLog.children.length > 100) {
    eventLog.removeChild(eventLog.lastChild);
  }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', init);