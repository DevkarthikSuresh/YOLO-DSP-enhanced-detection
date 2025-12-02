import cv2
import numpy as np
from ultralytics import YOLO

def load_model(weights_path):
    """
    Load YOLO model using the given weights.
    """
    return YOLO(weights_path)

def run_detect(model, img_bgr, conf=0.25, iou=0.5):
    """
    Run YOLO detection on an image and return:
        - detections as dictionaries
        - annotated image (BGR)
    """
    result = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        verbose=False
    )[0]

    # Extract detections
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
    else:
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))
        classes = np.zeros((0,), int)

    names = result.names
    detections = []

    for box, score, cls in zip(boxes, scores, classes):
        detections.append({
            "label": names[cls],
            "score": float(score),
            "box_xyxy": box.tolist()
        })

    # YOLO built-in visualization
    annotated = result.plot()  # returns BGR overlay

    return detections, annotated
