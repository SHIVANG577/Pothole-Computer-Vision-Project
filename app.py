import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import time

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

# ---------------- TITLE ---------------- #
st.title("🛣️ AI-Based Pothole Detection")
st.write("Detect potholes in road images using a fine-tuned YOLOv12 model.")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.30,
    0.05
)
mode = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)
if mode == "Image":

    uploaded_file = st.sidebar.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        results = model.predict(
            image_np,
            conf=confidence,
            verbose=False
        )

        annotated = results[0].plot()

        detections = len(results[0].boxes)

        confidences = []

        for box in results[0].boxes:
            confidences.append(float(box.conf))

        avg_conf = np.mean(confidences) if confidences else 0
        max_conf = np.max(confidences) if confidences else 0

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("Detected Potholes", detections)
        c2.metric("Average Confidence", f"{avg_conf:.2f}")
        c3.metric("Highest Confidence", f"{max_conf:.2f}")

  # ---------------- VIDEO MODE ---------------- #

elif mode == "Video":

    uploaded_video = st.sidebar.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("Original Video")
        st.video(uploaded_video)

        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        frame_placeholder = st.empty()

        c1, c2, c3 = st.columns(3)

        fps_metric = c1.empty()
        frame_metric = c2.empty()
        detection_metric = c3.empty()

        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        prev_time = time.time()

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            results = model.predict(
                frame,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            detections = len(results[0].boxes)

            current_time = time.time()

            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            fps_metric.metric(
                "FPS",
                f"{fps:.2f}"
            )

            frame_metric.metric(
                "Frame",
                f"{frame_number}/{total_frames}"
            )

            detection_metric.metric(
                "Detections",
                detections
            )

            progress_bar.progress(frame_number / total_frames)

            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        progress_bar.empty()

        st.success("✅ Video Processing Completed!")