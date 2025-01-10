import os
import cv2

def draw_detections(frame, detections, labels):
    for class_id, box, confidence in detections:
        x, y, w, h = box
        label = f"{labels[class_id]}: {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def save_cropped_images(frame, detections, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (_, box, _) in enumerate(detections):
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w > 0 and h > 0:
            cropped = frame[y:y+h, x:x+w]
            if cropped.size != 0:
                output_path = os.path.join(output_dir, f"object_{i}.jpg")
                cv2.imwrite(output_path, cropped)

def format_detections_as_json(frame_index, detections, labels):
    formatted = []
    for i, (class_id, box, confidence) in enumerate(detections):
        x, y, w, h = box
        entry = {
            "frame": frame_index,
            "object": labels[class_id],
            "id": i,
            "bbox": [x, y, x + w, y + h],
            "confidence": round(confidence, 2),
            "subobject": None  # Placeholder for potential nested sub-objects
        }
        formatted.append(entry)
    return formatted
