import cv2

class ObjectDetector:
    def __init__(self, model_path, config_path, labels_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        self.labels = open(labels_path).read().strip().split("\n")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_objects(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    box = detection[:4] * [w, h, w, h]
                    center_x, center_y, width, height = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        return [(class_ids[i], boxes[i], confidences[i]) for i in indices.flatten()]
