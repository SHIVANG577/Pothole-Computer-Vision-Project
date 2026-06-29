import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
import time
import io

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Based Pothole Detection",
    page_icon="🛣",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#F7F9FC;
}

.block-container{
    padding-top:1rem;
}

.metric-container{
    background:white;
    border-radius:12px;
    padding:10px;
}

div[data-testid="metric-container"]{
    border-radius:12px;
    border:1px solid #E5E7EB;
    padding:15px;
    background:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🛣 AI-Based Pothole Detection using YOLOv12")

st.markdown("""
Detect potholes from **road images and videos**
using a fine-tuned **YOLOv12 Deep Learning model**.

Upload an image or video and obtain:

- ✅ Automatic pothole detection
- ✅ Confidence score
- ✅ Bounding box dimensions
- ✅ Severity classification
- ✅ Detection analytics
""")

st.divider()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

with st.spinner("Loading YOLOv12 Model..."):

    model = load_model()

st.success("Model Loaded Successfully")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("⚙ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.30,
    0.05
)

mode = st.sidebar.radio(
    "Choose Input",
    [
        "Image",
        "Video"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Model : YOLOv12

Dataset : Custom Pothole Dataset

Framework : Ultralytics

Inference : PyTorch
"""
)

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def get_severity(area):

    if area < 5000:
        return "🟢 Small"

    elif area < 20000:
        return "🟡 Medium"

    else:
        return "🔴 Large"


def image_download(image):

    _, buffer = cv2.imencode(".jpg", image)

    return buffer.tobytes()


def draw_summary(df):

    small = len(df[df["Severity"]=="🟢 Small"])
    medium = len(df[df["Severity"]=="🟡 Medium"])
    large = len(df[df["Severity"]=="🔴 Large"])

    c1,c2,c3=st.columns(3)

    c1.metric("🟢 Small",small)

    c2.metric("🟡 Medium",medium)

    c3.metric("🔴 Large",large)

# ============================================================
# IMAGE MODE
# ============================================================

if mode == "Image":

    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload Road Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_np = np.array(image)

        start = time.time()

        results = model.predict(
            image_np,
            conf=confidence,
            verbose=False
        )

        inference_time = time.time() - start

        result = results[0]

        annotated = result.plot()

        detections = len(result.boxes)

        image_height, image_width = image_np.shape[:2]

        rows = []

        confidences = []

        largest_area = 0

        for idx, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            width = int(x2 - x1)

            height = int(y2 - y1)

            area = width * height

            confidence_score = float(box.conf)

            confidences.append(confidence_score)

            severity = get_severity(area)

            if area > largest_area:
                largest_area = area

            rows.append({

                "ID": idx + 1,
                "Confidence": round(confidence_score,3),
                "Width (px)": width,
                "Height (px)": height,
                "Area (px²)": area,
                "Severity": severity

            })

        avg_conf = np.mean(confidences) if confidences else 0

        max_conf = np.max(confidences) if confidences else 0

        # --------------------------------------------------
        # Images
        # --------------------------------------------------

        left,right=st.columns(2)

        with left:

            st.subheader("📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with right:

            st.subheader("🎯 Detection Result")

            st.image(
                annotated,
                use_container_width=True
            )

        st.divider()

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        c1,c2,c3,c4,c5,c6=st.columns(6)

        c1.metric(
            "Detected",
            detections
        )

        c2.metric(
            "Average Conf",
            f"{avg_conf:.2f}"
        )

        c3.metric(
            "Highest Conf",
            f"{max_conf:.2f}"
        )

        c4.metric(
            "Largest Area",
            f"{largest_area}"
        )

        c5.metric(
            "Resolution",
            f"{image_width}×{image_height}"
        )

        c6.metric(
            "Inference",
            f"{inference_time:.3f}s"
        )

        st.divider()

        # --------------------------------------------------
        # TABLE
        # --------------------------------------------------

        st.subheader("📋 Detection Details")

        df = pd.DataFrame(rows)

        if len(df):

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No potholes detected.")

        st.divider()

        # --------------------------------------------------
        # Severity
        # --------------------------------------------------

        st.subheader("🚦 Severity Distribution")

        if len(df):

            draw_summary(df)

        st.divider()

        # --------------------------------------------------
        # Charts
        # --------------------------------------------------

        if len(df):

            chart1,chart2=st.columns(2)

            with chart1:

                st.subheader("📈 Confidence")

                st.bar_chart(
                    df.set_index("ID")["Confidence"]
                )

            with chart2:

                st.subheader("📊 Severity")

                severity_counts=df["Severity"].value_counts()

                st.bar_chart(severity_counts)

        st.divider()

        # --------------------------------------------------
        # Download
        # --------------------------------------------------

        st.download_button(

            "⬇ Download Annotated Image",

            data=image_download(annotated),

            file_name="pothole_detection.jpg",

            mime="image/jpeg"

        )

        st.divider()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        st.success("Detection Completed Successfully")

        st.markdown("### 📄 Detection Summary")

        st.markdown(f"""
**Total Potholes:** {detections}

**Average Confidence:** {avg_conf:.2f}

**Largest Pothole Area:** {largest_area} px²

**Image Resolution:** {image_width} × {image_height}

**Processing Time:** {inference_time:.3f} sec
""")
        
# ============================================================
# VIDEO MODE
# ============================================================

elif mode == "Video":

    uploaded_video = st.sidebar.file_uploader(
        "📤 Upload Road Video",
        type=["mp4", "avi", "mov"]
    )

    process = st.sidebar.button("🚀 Start Detection")

    if uploaded_video is not None:

        st.video(uploaded_video)

    if uploaded_video is not None and process:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = "processed_output.mp4"

        writer = cv2.VideoWriter(

            output_path,

            cv2.VideoWriter_fourcc(*"mp4v"),

            fps,

            (width, height)

        )

        st.divider()

        progress = st.progress(0)

        status = st.empty()

        metric1, metric2, metric3, metric4 = st.columns(4)

        fps_metric = metric1.empty()

        frame_metric = metric2.empty()

        detection_metric = metric3.empty()

        time_metric = metric4.empty()

        frame_number = 0

        total_detections = 0

        largest_area = 0

        confidence_list = []

        severity_count = {
            "Small": 0,
            "Medium": 0,
            "Large": 0
        }

        start = time.time()

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

            result = results[0]

            annotated = result.plot()

            detections = len(result.boxes)

            total_detections += detections

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                w = int(x2 - x1)

                h = int(y2 - y1)

                area = w * h

                largest_area = max(
                    largest_area,
                    area
                )

                confidence_list.append(
                    float(box.conf)
                )

                if area < 5000:

                    severity_count["Small"] += 1

                elif area < 20000:

                    severity_count["Medium"] += 1

                else:

                    severity_count["Large"] += 1

            writer.write(annotated)

            elapsed = time.time() - start

            current_fps = frame_number / elapsed

            progress.progress(frame_number / total_frames)

            status.info(
                f"Processing Frame {frame_number} / {total_frames}"
            )

            fps_metric.metric(
                "⚡ FPS",
                f"{current_fps:.2f}"
            )

            frame_metric.metric(
                "🎞 Frame",
                frame_number
            )

            detection_metric.metric(
                "🎯 Detections",
                total_detections
            )

            time_metric.metric(
                "⏱ Time",
                f"{elapsed:.1f}s"
            )

        cap.release()

        writer.release()

        progress.empty()

        status.empty()

        average_conf = (
            np.mean(confidence_list)
            if confidence_list else 0
        )

        st.success("✅ Video Processing Completed")

        st.divider()

        st.subheader("📹 Processed Video")

        st.video(output_path)

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Detections",
            total_detections
        )

        c2.metric(
            "Average Confidence",
            f"{average_conf:.2f}"
        )

        c3.metric(
            "Largest Pothole",
            f"{largest_area} px²"
        )

        c4.metric(
            "Resolution",
            f"{width}×{height}"
        )

        st.divider()

        severity_df = pd.DataFrame({

            "Severity": severity_count.keys(),

            "Count": severity_count.values()

        })

        left, right = st.columns(2)

        with left:

            st.subheader("🚦 Severity Distribution")

            st.bar_chart(
                severity_df.set_index("Severity")
            )

        with right:

            st.subheader("📈 Processing Summary")

            summary = pd.DataFrame({

                "Metric": [

                    "Frames",

                    "Detections",

                    "Average Confidence",

                    "Largest Area"

                ],

                "Value": [

                    total_frames,

                    total_detections,

                    round(average_conf, 2),

                    largest_area

                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        with open(output_path, "rb") as file:

            st.download_button(

                "⬇ Download Processed Video",

                file,

                file_name="pothole_detection_output.mp4",

                mime="video/mp4"

            )