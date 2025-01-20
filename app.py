import eventlet

eventlet.monkey_patch()

from ultralytics import YOLO
from fer import FER
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Initialize YOLO and FER models
yolo_model = YOLO("yolov8n.pt")
emotion_detector = FER(mtcnn=True)

# Thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.45
EMOTION_CONFIDENCE_THRESHOLD = 0.4
BRIGHTNESS_THRESHOLD = 30


@app.route('/')
def index():
    return render_template("index.html")


def check_frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    return average_brightness > BRIGHTNESS_THRESHOLD


def process_frame(frame):
    try:
        if not check_frame_brightness(frame):
            return 0, 0

        # Run YOLO detection
        results = yolo_model(frame, conf=PERSON_CONFIDENCE_THRESHOLD)

        if len(results) == 0:
            return 0, 0

        user_count = 0
        happy_count = 0

        result = results[0]
        person_boxes = [box for box in result.boxes if int(box.cls) == 0]
        user_count = len(person_boxes)

        if user_count == 0:
            return 0, 0

        # Process emotions for detected people
        for box in person_boxes:
            try:
                conf = float(box.conf[0])
                if conf < PERSON_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if x2 <= x1 or y2 <= y1:
                    continue

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                # Resize if the crop is too large
                max_dimension = 300
                h, w = person_crop.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    person_crop = cv2.resize(person_crop, (new_w, new_h))

                emotions = emotion_detector.detect_emotions(person_crop)

                if emotions and len(emotions) > 0:

                    emotion_scores = emotions[0]['emotions']
                    happy_score = emotion_scores['happy']

                    if happy_score > EMOTION_CONFIDENCE_THRESHOLD:
                        happy_count += 1

            except Exception as e:
                print(f"Error processing person box: {e}")
                continue

        return user_count, happy_count

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return 0, 0


@socketio.on("stream_frame")
def handle_stream_frame(data):
    try:
        frame_data = data.split(",")[1]
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            socketio.emit("update_counts", {"user_count": 0, "happy_count": 0})
            return

        user_count, happy_count = process_frame(frame)

        socketio.emit("update_counts", {
            "user_count": int(user_count),
            "happy_count": int(happy_count)
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        socketio.emit("update_counts", {"user_count": 0, "happy_count": 0})



if __name__ == "__main__":
    port = int(os.environ.get("PORT"))
    print(f"Starting server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port)

    
    
