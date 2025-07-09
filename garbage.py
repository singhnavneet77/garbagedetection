import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2 
from ultralytics import YOLO
import av 
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()
database_url = os.getenv("MONGODB_API_KEY")

# ------------------ MongoDB Connection ------------------
MONGO_URI = database_url
client = MongoClient(MONGO_URI)
db = client["garbage_db"]
collection = db["classified_garbage"]

# ------------------ YOLO Model ------------------
model = YOLO("best2.pt")

# ------------------ Streamlit UI ------------------
st.title("🗑️ Real-Time Garbage Detection with YOLOv8")
st.markdown("This app uses your webcam to detect garbage using a custom-trained YOLOv8 model.")

# Sidebar
with st.sidebar:
    view = st.button("🧾 View Classified Garbage")
    clear = st.button("🗑️ Clear Data")

    if clear:
        collection.delete_many({})
        st.success("All classified garbage data has been cleared.")

# ------------------ Video Transformer ------------------
class YOLODetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model(img)
        boxes = results[0].boxes

        # Save to MongoDB
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = results[0].names[cls_id]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc = {
                "Garbage Type": label,
                "Time": timestamp
            }
            collection.insert_one(doc)

        # Annotated frame
        annotated_frame = results[0].plot()
        return annotated_frame

# ------------------ Start Stream ------------------
webrtc_streamer(
    key="yolo-garbage",
    video_processor_factory=YOLODetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ------------------ Display Table ------------------
if view:
    st.subheader("📋 Classified Garbage Table")
    data = list(collection.find({}, {"_id": 0}))
    if data:
        st.table(data)
    else:
        st.info("No garbage classified yet.")
