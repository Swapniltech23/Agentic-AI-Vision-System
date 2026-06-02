# Agentic AI Vision System

## Overview

Agentic AI Vision System is a multimodal computer vision application designed to assist users through real-time scene understanding, navigation support, and security monitoring. The system combines object detection, intelligent decision-making, voice feedback, and alert mechanisms to create an interactive AI-powered vision assistant.

This project was developed as a final-year Data Science major project.

---

## Key Features

### Navigation Assistance

* Detects obstacles and surrounding objects in real time.
* Provides voice guidance for safer movement.
* Warns users when objects are too close.

### Security Monitoring

* Monitors the environment continuously.
* Detects persons and important objects.
* Generates alerts when predefined security conditions are met.

### Real-Time Object Detection

* Powered by YOLOv8.
* Fast and accurate object recognition.
* Supports multiple object categories.

### Voice Feedback System

* Converts AI decisions into speech.
* Provides hands-free interaction.
* Delivers real-time alerts and notifications.

### Interactive Dashboard

* Built using Streamlit.
* User-friendly interface.
* Supports multiple operating modes.

---

## System Architecture

Camera Input → YOLOv8 Detection → Agentic Decision Layer → Navigation/Security Logic → Voice Feedback & Alerts

---

## Technologies Used

* Python
* YOLOv8
* OpenCV
* PyTorch
* Streamlit
* Text-to-Speech (TTS)

---

## Project Structure

```text
AI_VISION_SYSTEM/
│
├── app.py
├── stream.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Agentic-AI-Vision-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run stream.py
```

---

## Applications

* Assistive technology for visually impaired users.
* Smart surveillance systems.
* Indoor navigation support.
* AI-powered monitoring solutions.
* Educational demonstration of multimodal AI systems.

---

## Future Enhancements

* Face recognition integration.
* Cloud-based monitoring dashboard.
* Mobile application support.
* Advanced activity recognition.
* Multi-camera support.

---

## Team Members

* Mamun
* Team Member 2

---

## Disclaimer

This project was developed for educational and research purposes. The system demonstrates the integration of computer vision, AI reasoning, and voice interaction within a unified framework.
