import cv2
from src.object_detector import ObjectDetector
from src.utils import draw_detections, save_cropped_images, format_detections_as_json
import os
import json

class Detector:
    def __init__(self, model_path, config_path, labels_path, video_path, output_dir, output_json_file):
        self.detector = ObjectDetector(model_path, config_path, labels_path)
        self.video_path = video_path
        self.output_dir = output_dir
        self.output_json_file = output_json_file

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self):
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        detection_results = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection on the frame
            detections = self.detector.detect_objects(frame)

            # Draw bounding boxes and labels on the frame
            frame = draw_detections(frame, detections, self.detector.labels)

            # Save cropped images of detected objects
            save_cropped_images(frame, detections, self.output_dir)

            # Format detections as JSON
            frame_results = format_detections_as_json(frame_index, detections, self.detector.labels)
            detection_results.extend(frame_results)

            # Display the frame (optional, for visualization)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

        # Release video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Save detection results to JSON file
        with open(self.output_json_file, "w") as json_file:
            json.dump(detection_results, json_file, indent=4)

        print(f"Detections saved to {self.output_json_file}")
