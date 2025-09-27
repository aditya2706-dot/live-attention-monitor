# Live Attention Monitoring System

A real-time system that uses a webcam to detect faces and classify a person's attention level using Deep Learning and Computer Vision.

The system classifies the user's state into one of four categories:
- **🟢 Attentive**: User is looking towards the camera.
- **🟡 Distracted**: User's face is present but turned away.
- **🔴 Sleeping**: User's eyes are closed for a sustained period.
- **⚫ Not Present**: No face is detected in the camera feed.

## 🚀 Features

- **Live Webcam Feed**: Real-time video stream with face detection and attention status overlay.
- **Real-time Analytics**: A dynamic line chart showing the distribution of attention states over time.
- **Session Summary**: Pie chart and statistics summarizing the percentage of time spent in each state.
- **Data Logging**: All attention events are logged with timestamps.
- **Export to CSV**: Download the complete session log for further analysis.

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