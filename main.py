import os
import cv2
import json
from src.utils import draw_detections, save_cropped_images, format_detections_as_json
from src.object_detector import ObjectDetector

# Paths
VIDEO_PATH = "data/test_video.mp4"
MODEL_PATH = "models/yolov4.weights"
CONFIG_PATH = "models/yolov4.cfg"
LABELS_PATH = "models/coco.names"
OUTPUT_IMAGES_DIR = "outputs/cropped_images"
OUTPUT_JSON_FILE = "outputs/detection_output.json"

# Initialize Object Detector
detector = ObjectDetector(MODEL_PATH, CONFIG_PATH, LABELS_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# Create output directories if not exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# Process video
detection_results = []
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detector.detect_objects(frame)

    # Draw detections on the frame
    frame = draw_detections(frame, detections, detector.labels)

    # Save cropped images of detected objects
    save_cropped_images(frame, detections, OUTPUT_IMAGES_DIR)

    # Format detections for JSON
    frame_results = format_detections_as_json(frame_index, detections, detector.labels)
    detection_results.extend(frame_results)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save detections to JSON
with open(OUTPUT_JSON_FILE, "w") as json_file:
    json.dump(detection_results, json_file, indent=4)

print(f"Detections saved to {OUTPUT_JSON_FILE}")
