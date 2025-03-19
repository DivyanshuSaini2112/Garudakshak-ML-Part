# Garudakshak Anti-Drone Detection System

<div align="center">
  <img src="https://github.com/user-attachments/assets/002fa321-eb6a-4393-80e9-889c18d9a161" alt="Garudakshak Logo" width="600">
  
  *Advanced drone detection and neutralization system*
  
  [![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/DivyanshuSaini2112/Garudakshak-ML-Part?style=social)](https://github.com/DivyanshuSaini2112/Garudakshak-ML-Part/stargazers)
</div>

## 📋 Overview

Garudakshak is a state-of-the-art anti-drone detection and tracking system designed to identify, monitor, and provide real-time analytics for unauthorized drone activity. The system leverages advanced computer vision techniques, depth estimation algorithms, and GPS tracking to create a comprehensive drone detection and monitoring solution.

## ✨ Key Features

<table>
  <tr>
    <td>
      <ul>
        <li>🎯 <b>Real-time drone detection and tracking</b></li>
        <li>🔒 <b>Target locking mechanism</b> with visual indicators</li>
        <li>📏 <b>Depth estimation</b> using MiDaS deep learning model</li>
        <li>📍 <b>GPS position triangulation</b> for target location</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>🔄 <b>Persistent tracking</b> across video frames</li>
        <li>📊 <b>Comprehensive analytics dashboard</b></li>
        <li>⚡ <b>High-performance processing</b> optimized for real-time operation</li>
        <li>📱 <b>Multi-platform support</b></li>
      </ul>
    </td>
  </tr>
</table>

## 📊 Analytics Dashboard

The system includes a powerful analytics dashboard providing:

- 🗺️ GPS tracking path visualization
- 📏 Target distance monitoring
- 🚀 Speed/velocity analysis
- 🔒 Lock status tracking
- 📈 Statistical data analysis

## 🖥️ System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- Webcam or video input device
- Required Python packages:

```
opencv-python>=4.5.0
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
matplotlib>=3.3.0
geopy>=2.0.0
pillow>=8.0.0
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DivyanshuSaini2112/Garudakshak-ML-Part.git
   cd Garudakshak-ML-Part
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required model weights:**
   ```bash
   # The system will automatically download MiDaS weights on first run
   ```

## 🎮 Usage

Run the main detection script:
```bash
python Detect.py
```

### Controls
- Press 'q' to exit the detection system and view the analytics dashboard

## 🏗️ System Architecture

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=Garudakshak+System+Architecture" alt="System Architecture" width="700">
</div>

The Garudakshak system consists of several key components:

1. **Detection Module**: Uses Haar cascade classifiers for initial drone detection
2. **Depth Estimation**: Implements MiDaS deep learning model to estimate target distance
3. **Position Triangulation**: Calculates GPS coordinates based on camera position and target offsets
4. **Tracking System**: Maintains persistent identification of targets across frames
5. **Analytics Dashboard**: Provides comprehensive visualization of tracking data

## ⚙️ Configuration

The system includes configurable parameters in the code:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detection | 0.35 |
| `TARGET_LOCKED_THRESHOLD` | Frames required for target lock | 5 |
| `GPS_UPDATE_INTERVAL` | Seconds between GPS position updates | 1.0 |
| `DISPLAY_WIDTH/HEIGHT` | Resolution for processing and display | 1280x720 |

## 📚 Documentation

For more detailed information, please refer to the following documentation:

- [Garudakshak ML Part](docs/Garudakshak_ML_part.pdf)
- [Drone Neutralization Methods](docs/Drone_Neutralization_Methods.pdf)
- [Legal Guidelines](docs/Garudakshak_legal_guidelines.docx)
- [Bill of Materials](docs/Bill_Of_Material_for_Drone_Neutralization.pdf)

## 📂 Project Structure

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

## 🤝 Contributing

Contributions to the Garudakshak project are welcome! Please feel free to submit a Pull Request.

Guidelines for contributing:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Intel for the MiDaS depth estimation model
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework

---

<div align="center">
  <p>Developed by <a href="https://github.com/DivyanshuSaini2112">Divyanshu Saini</a> and contributors</p>
  <p>© 2023-2025 Garudakshak Project</p>
</div>
