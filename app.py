import streamlit as st
import cv2
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

from capture import classify_attention, COLORS

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Attention Monitoring",
    page_icon="🧠",
    layout="wide"
)

st.title("Live Attention Monitoring System 🧠")
st.write("This application uses your webcam to monitor your attention level in real-time.")

# --- Constants and Initializations ---
LOG_FILE = "data/attention_log.csv"

# Initialize the log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    pd.DataFrame(columns=["timestamp", "status", "confidence"]).to_csv(LOG_FILE, index=False)

# --- Session State Initialization ---
if 'run' not in st.session_state:
    st.session_state.run = False
    # On first run, clear the old log file
# Define dtypes to prevent Pandas FutureWarning
log_dtypes = {
    "timestamp": "datetime64[ns]",
    "status": "object",
    "confidence": "float64"
}
if 'log_df' not in st.session_state:
    # Initialize an empty DataFrame with correct dtypes if the log is empty or doesn't exist
    st.session_state.log_df = pd.DataFrame(columns=log_dtypes.keys()).astype(log_dtypes)
if 'capture' not in st.session_state:
    st.session_state.capture = None
if 'state' not in st.session_state:
    st.session_state.state = {
        'eyes_closed_start_time': None,
        'eyes_closed_threshold_seconds': 2.0, # 2 seconds
        'last_chart_update_time': 0,
        'chart_update_interval': 1.0, # 1 second
        'session_start_time': None
    }

def start_camera():
    """Initializes the webcam capture."""
    # Prevent re-initializing if already running
    if not st.session_state.run:
        st.session_state.capture = cv2.VideoCapture(0)
        if not st.session_state.capture.isOpened():
            st.error("Cannot open webcam. Please check permissions and connections.")
            st.session_state.run = False
            return
        st.session_state.state['session_start_time'] = time.time()
        st.session_state.run = True

def stop_camera():
    """Releases the webcam and stops the loop."""
    if st.session_state.run:
        # Save the final log before stopping
        if not st.session_state.log_df.empty:
            st.session_state.log_df.to_csv(LOG_FILE, index=False)

        st.session_state.run = False
        if st.session_state.capture:
            st.session_state.capture.release()
        st.session_state.capture = None
        st.rerun() # Force a rerun to update the UI instantly

def reset_session():
    """Clears the log and resets the session."""
    stop_camera()
    # Clear the DataFrame in session state and the log file
    st.session_state.log_df = pd.DataFrame(columns=log_dtypes.keys()).astype(log_dtypes)
    st.session_state.log_df.to_csv(LOG_FILE, index=False)
    st.rerun()

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Live Camera Feed")
    start_button = st.button("Start Camera", on_click=start_camera)
    stop_button = st.button("Stop Camera", on_click=stop_camera)
    FRAME_WINDOW = st.image([])

with col2:
    st.header("Real-time Analytics")
    
    # Placeholders for metrics and charts
    kpi_cols = st.columns(3)
    kpi_placeholders = [kpi_cols[i].empty() for i in range(3)]
    
    st.markdown("---")
    pie_chart_placeholder = st.empty()
    summary_table_placeholder = st.empty()
    time_series_placeholder = st.empty()

with st.spinner('Loading models and initializing...'):
    classify_attention # This will trigger the model loading in capture.py

# --- Main Application Loop ---
while st.session_state.run and st.session_state.capture:
    ret, frame = st.session_state.capture.read()
    if not ret:
        st.warning("Failed to capture frame from webcam. Stopping...")
        stop_camera()
        st.rerun()
        break

    # Process the frame for attention classification
    processed_frame, label, confidence, new_state = classify_attention(frame, st.session_state.state)
    st.session_state.state = new_state # Update state

    # Display the processed frame
    FRAME_WINDOW.image(processed_frame, channels="BGR")

    # --- Logging ---
    new_log = pd.DataFrame({
        "timestamp": [datetime.now()],
        "status": [label],
        "confidence": [confidence]
    })
    st.session_state.log_df = pd.concat([st.session_state.log_df, new_log], ignore_index=True)

    # --- Update Analytics ---
    current_time = time.time()
    if (st.session_state.state['session_start_time'] is not None and
        current_time - st.session_state.state['last_chart_update_time'] > st.session_state.state['chart_update_interval']):
        
        st.session_state.state['last_chart_update_time'] = current_time
        log_df = st.session_state.log_df
        
        if not log_df.empty:
            # --- KPIs ---
            session_duration = time.time() - st.session_state.state['session_start_time']
            kpi_placeholders[0].metric("Session Duration", f"{session_duration:.1f}s")
            kpi_placeholders[1].metric("Current Status", label)
            attentive_percentage = (log_df['status'] == 'Attentive').mean() * 100
            kpi_placeholders[2].metric("Attentive", f"{attentive_percentage:.1f}%")

            # Pie Chart for status distribution
            status_counts = log_df['status'].value_counts()
            
            # Convert BGR (0-255) colors from OpenCV to RGB (0-1) for Matplotlib
            chart_colors = []
            for status in status_counts.index:
                bgr_color = COLORS.get(status, (128, 128, 128)) # Default to gray
                # Convert BGR to RGB and normalize to 0-1 range
                rgb_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
                chart_colors.append(rgb_normalized)

            fig1, ax1 = plt.subplots(figsize=(5, 3))
            ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=chart_colors)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            pie_chart_placeholder.pyplot(fig1)
            plt.close(fig1) # Explicitly close the figure to free memory

            # --- Summary Table ---
            total_frames = len(log_df)
            summary_data = []
            for status, count in status_counts.items():
                percentage = (count / total_frames) * 100
                summary_data.append({"Status": status, "Percentage": f"{percentage:.1f}%", "Frames": count})
            summary_df = pd.DataFrame(summary_data)
            summary_table_placeholder.dataframe(summary_df, width='stretch')

            # Time Series Chart
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            # Create a numeric representation for status for plotting
            status_map = {"Attentive": 3, "Distracted": 2, "Sleeping": 1, "Not Present": 0}
            log_df['status_numeric'] = log_df['status'].map(status_map)
            
            fig2, ax2 = plt.subplots(figsize=(7, 2.5))
            ax2.plot(log_df['timestamp'], log_df['status_numeric'], marker='o', linestyle='-')
            plt.yticks(list(status_map.values()), list(status_map.keys()))
            plt.title("Attention Level Over Time")
            plt.xlabel("Time")
            plt.ylabel("Status")
            plt.grid(True)
            fig2.tight_layout()
            time_series_placeholder.pyplot(fig2)
            plt.close(fig2) # Explicitly close the figure to free memory

# --- After the loop finishes ---
if not st.session_state.run:
    st.info("Camera is off. Click 'Start Camera' to begin monitoring.")

# --- Data Export Section ---
st.sidebar.header("Session Log")
if not st.session_state.log_df.empty:
    st.sidebar.button("Reset Session", on_click=reset_session, type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.write("Most recent events:")
    st.sidebar.dataframe(st.session_state.log_df.tail())

    # Convert DataFrame to CSV for download
    csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="Download Session Log as CSV",
        data=csv,
        file_name='attention_log.csv',
        mime='text/csv',
    )
else:
    st.sidebar.info("No data logged yet.")

with st.sidebar.expander("About this App"):
    st.markdown(
        """
        This app provides real-time attention monitoring using your webcam.
        - **🟢 Attentive**: Looking at the screen.
        - **🟡 Distracted**: Looking away.
        - **🔴 Sleeping**: Eyes closed for a sustained period.
        - **⚫ Not Present**: No face detected.
        """
    )