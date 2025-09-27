# Live Attention Monitoring System

An intelligent, real-time system that leverages deep learning and computer vision to analyze a user's attention level through a standard webcam. Built with PyTorch, OpenCV, and Streamlit, this project provides a live dashboard with insightful analytics on user engagement.

## Overview

The application monitors a user's video feed to classify their state into four distinct categories:
- **🟢 Attentive**: The user is looking towards the camera, indicating focus.
- **🟡 Distracted**: The user's head is turned away, suggesting a lapse in attention.
- **🔴 Sleeping**: The user's eyes have been closed for a sustained period.
- **⚫ Not Present**: No face is detected in the frame.

This classification is performed in real-time, with results displayed on an interactive dashboard that includes live metrics, session summaries, and data logging capabilities.

## 💡 How It Works

The system follows a multi-stage computer vision pipeline for each frame captured from the webcam:
1.  **Face Detection**: It first uses OpenCV's Haar Cascade classifier to locate a face in the video frame.
2.  **Eye Detection**: Within the detected face region, it searches for eyes. The presence (or absence) of eyes is a key heuristic.
3.  **Hybrid Classification**:
    -   If at least one eye is detected, the system confidently classifies the user as **Attentive**. This provides a stable and reliable baseline.
    -   If no eyes are detected, the situation is ambiguous. The system then uses a custom-trained PyTorch CNN model to classify the face as either **Distracted** or potentially part of a "Sleeping" state.
    -   If no face is found, the state is **Not Present**.
4.  **Dashboarding**: All data is streamed to a Streamlit dashboard, which visualizes the attention status, provides real-time analytics, and allows for data export.

## � Features

- **Live Webcam Feed**: Real-time video stream with face detection and attention status overlay.
- **Real-time Analytics**: A dynamic line chart showing the distribution of attention states over time.
- **Session Summary**: Pie chart and statistics summarizing the percentage of time spent in each state.
- **Data Logging**: All attention events are logged with timestamps.
- **Export to CSV**: Download the complete session log for further analysis.
- **Hybrid Intelligence**: Combines robust heuristics (eye detection) with a deep learning model (CNN) for accurate and stable classification.

## 🧠 Tech Stack

- **Python 3.x**
- **Streamlit**: For the interactive web dashboard.
- **PyTorch**: For the deep learning classification model.
- **OpenCV**: For webcam access, face/eye detection, and image processing.
- **Pandas & NumPy**: For data manipulation and logging.
- **Matplotlib**: For plotting analytics charts.

## ⚙️ Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd attention_monitor
    ```

2.  **Install dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 💻 How to Run

1.  Navigate to the project's root directory (`attention_monitor/`).
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your default web browser. Click "Start Camera" to begin monitoring.