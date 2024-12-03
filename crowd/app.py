from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import threading

app = Flask(__name__)

# Load YOLO model and configuration
weights_path = "yolov4.weights"
config_path = "yolov4.cfg"
coco_names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera and threading variables
camera = cv2.VideoCapture(0)  # Laptop's default camera
frame_lock = threading.Lock()
frame = None

# Thread to capture frames from the camera
def capture_frames():
    global frame
    while True:
        ret, new_frame = camera.read()
        if ret:
            with frame_lock:
                frame = new_frame
        else:
            break

capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Process video frame to detect crowd density
def process_frame():
    global frame
    with frame_lock:
        if frame is None:
            return None, 0

        # Resize the frame for YOLO processing
        resized_frame = cv2.resize(frame, (640, 480))
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Analyze detections
        height, width, _ = resized_frame.shape
        people_count = 0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "person":
                    people_count += 1

        return resized_frame, people_count

# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    def generate_feed():
        while True:
            processed_frame, _ = process_frame()
            if processed_frame is not None:
                _, buffer = cv2.imencode('.jpg', processed_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to get real-time crowd density
@app.route('/get_density')
def get_density():
    _, people_count = process_frame()

    # Define density thresholds
    if people_count < 2:
        density = "Low"
    elif 2 <= people_count <= 4:
        density = "Medium"
    else:
        density = "High"

    data = {
        "locations": [
            {"name": "Canteen", "density": density, "people": people_count},
            {"name": "Library", "density": "Low", "people": 2}  # Static example data
        ]
    }
    return jsonify(data)

# Dashboard endpoint
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
