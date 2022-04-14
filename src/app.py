import fire
import cv2
import time

from object_detection import ObjectDetectionModel, YoloV4Model
from tracking import MultiTracker

TRACKING_ALGORITHMS = {
    "t": "tld",
    "m": "mosse",
    "k": "kcf",
    "b": "boosting",
    "c": "csrt",
    "i": "mil",
    "f": "medianflow",
}


def main(
    source: str = 0,
    weights_file: str = "src/model/yolov4-tiny.weights",
    cfg_file: str = "src/model/yolov4-tiny.cfg",
    labels_file: str = None,
    min_score: float = 0.6,
    nms_score: float = 0.5,
):
    tracking = False
    activate_tracking = False
    tracking_algorithm = "medianflow"

    yolov4_model = YoloV4Model(
        weights_file=weights_file, cfg_file=cfg_file, image_size=(416, 416)
    )
    detector = ObjectDetectionModel(
        model=yolov4_model,
        labels_file=labels_file,
        min_score=min_score,
        nms_score=nms_score,
    )

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not tracking:
            predictions = detector.predict(frame)
        else:
            start = time.time()
            bboxes = tracker.update(frame)
            duration = time.time() - start

            predictions["bboxes"] = bboxes
            predictions["inference_time"] = duration

        if activate_tracking and len(predictions["bboxes"]) > 0:
            tracker = MultiTracker(
                bboxes=predictions["bboxes"],
                frame=frame,
                tracker_type=tracking_algorithm,
            )
            tracking = True
            activate_tracking = False

        annotated_frame = detector.draw_predictions(frame, predictions)

        detector_state = f"Tracking: {tracking_algorithm}" if tracking else "Detecting"
        color = (0, 255, 0) if tracking else (0, 0, 255)

        annotated_frame = cv2.putText(
            annotated_frame, detector_state, (10, 80), 0, 1, color, 2
        )

        cv2.imshow("frame", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Quit
            break
        elif key == ord("d"):  # Detecting
            tracking = False

        if key != 255:
            tracking_algorithm = TRACKING_ALGORITHMS.get(chr(key), "")

            if tracking_algorithm != "":
                activate_tracking = True


if __name__ == "__main__":
    fire.Fire(main)
