# Garudakshak Anti-Drone Detection System

![image](https://github.com/user-attachments/assets/002fa321-eb6a-4393-80e9-889c18d9a161)


## Overview

Garudakshak is an advanced anti-drone detection and tracking system designed to identify, monitor, and provide real-time analytics for unauthorized drone activity. The system combines computer vision, depth estimation, and GPS tracking to create a comprehensive drone detection solution.

## Features

- **Real-time drone detection and tracking**
- **Target locking mechanism** with visual indicators
- **Depth estimation** using MiDaS deep learning model
- **GPS position triangulation** for target location
- **Persistent tracking** across video frames
- **Comprehensive analytics dashboard** with:
  - GPS tracking path visualization
  - Target distance monitoring
  - Speed/velocity analysis
  - Lock status tracking
- **High-performance processing** optimized for real-time operation

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- Webcam or video input device
- Required Python packages:
  - OpenCV
  - PyTorch
  - torchvision
  - NumPy
  - Matplotlib
  - Geopy
  - Pillow

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/DivyanshuSaini2112/Garudakshak-ML-Part.git
   cd Garudakshak-ML-Part
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download required model weights:
   ```
   # The system will automatically download MiDaS weights on first run
   ```

## Usage

Run the main detection script:
```
python Detect.py
```

### Controls
- Press 'q' to exit the detection system and view the analytics dashboard

## System Architecture

The Garudakshak system consists of several key components:

1. **Detection Module**: Uses Haar cascade classifiers for initial drone detection
2. **Depth Estimation**: Implements MiDaS deep learning model to estimate target distance
3. **Position Triangulation**: Calculates GPS coordinates based on camera position and target offsets
4. **Tracking System**: Maintains persistent identification of targets across frames
5. **Analytics Dashboard**: Provides comprehensive visualization of tracking data

## Configuration

The system includes configurable parameters in the code:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for detection (default: 0.35)
- `TARGET_LOCKED_THRESHOLD`: Frames required for target lock (default: 5)
- `GPS_UPDATE_INTERVAL`: Seconds between GPS position updates (default: 1.0)
- `DISPLAY_WIDTH/HEIGHT`: Resolution for processing and display

## Dashboard

The analytics dashboard provides comprehensive visualization of:

- Target GPS tracking path
- Real-time distance measurements
- Target velocity analysis
- Lock status history
- Summary statistics

## Documentation

For more detailed information, please refer to the following documentation:

- [Garudakshak ML Part](docs/Garudakshak_ML_part.pdf)
- [Drone Neutralization Methods](docs/Drone_Neutralization_Methods.pdf)
- [Legal Guidelines](docs/Garudakshak_legal_guidelines.docx)
- [Bill of Materials](docs/Bill_Of_Material_for_Drone_Neutralization.pdf)

## Project Structure

```
Garudakshak-ML-Part/
├── Detect.py               # Main detection script
├── Cords_Final.py          # GPS coordinate handling
├── Merger.py               # Data fusion module
├── Satvik.py               # Additional utilities
├── final_with_center_offset.py  # Center offset calculation
├── sideline.py             # Peripheral detection
├── test1.py - test5.py     # Test scripts
├── docs/                   # Documentation files
└── README.md               # This file
```

## Contributing

Contributions to the Garudakshak project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel for the MiDaS depth estimation model
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework
