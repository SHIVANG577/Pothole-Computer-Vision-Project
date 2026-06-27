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

    process = st.sidebar.button("🚀 Process Video")

    if uploaded_video is not None:

        st.video(uploaded_video)

    if uploaded_video is not None and process:

        # Save uploaded video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = "processed_output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress_bar = st.progress(0)

        status = st.empty()

        metric1, metric2, metric3 = st.columns(3)

        fps_metric = metric1.empty()
        frame_metric = metric2.empty()
        detection_metric = metric3.empty()

        frame = 0
        total_detections = 0

        start_time = time.time()

        while cap.isOpened():

            success, image = cap.read()

            if not success:
                break

            frame += 1

            results = model.predict(
                image,
                conf=confidence,
                verbose=False
            )

            annotated = results[0].plot()

            detections = len(results[0].boxes)

            total_detections += detections

            writer.write(annotated)

            elapsed = time.time() - start_time

            current_fps = frame / elapsed if elapsed > 0 else 0

            fps_metric.metric(
                "Processing FPS",
                f"{current_fps:.2f}"
            )

            frame_metric.metric(
                "Frame",
                f"{frame}/{total_frames}"
            )

            detection_metric.metric(
                "Current Detections",
                detections
            )

            progress_bar.progress(frame / total_frames)

            status.write(
                f"Processing frame {frame} of {total_frames}..."
            )

        cap.release()
        writer.release()

        progress_bar.empty()
        status.empty()

        st.success("✅ Video Processing Completed!")

        st.subheader("Processed Video")

        st.video(output_path)

        st.metric(
            "Total Pothole Detections",
            total_detections
        )

        with open(output_path, "rb") as file:

            st.download_button(
                "⬇ Download Processed Video",
                file,
                file_name="pothole_detection_output.mp4",
                mime="video/mp4"
            )