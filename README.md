# Object Detection System

## Overview
This project implements an object detection system using the YOLOv4 model, designed to work on CPU for portability and scalability. The system performs real-time detection (10–30 FPS) on videos, crops detected objects as individual images, and outputs detection details in a hierarchical JSON format.

---

## Features
- Real-time object detection using YOLOv4.
- Cropped images of detected objects saved to a designated directory.
- Hierarchical JSON outputs describing detected objects and their relationships.
- Modular design for easy extension and customization.

---

## Setup Instructions

### Prerequisites
- **Python 3.8 or higher**
- **Libraries:**
  - OpenCV
  - NumPy

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Anupam-75/Computer-Vision-System.git
   cd object-detection-system
