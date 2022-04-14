import numpy as np
from abc import ABC, abstractmethod
import cv2

from typing import Any, Tuple
import logging

logger = logging.getLogger(__name__)

bboxes = scores = categories = np.ndarray
image_height = image_width = int
prediction = Any


class AbstractDetectionModel(ABC):
    """Abstract model predictor with common predict method."""

    def __init__(self, weights_file: str, image_size: Tuple[int, int] = None):
        self._weights_file = weights_file
        self._image_size = image_size

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> prediction:
        ...

    @abstractmethod
    def normalize_predictions(
        self, image_shape: Tuple[image_height, image_width], predictions: np.ndarray
    ) -> Tuple[bboxes, scores, categories]:
        ...

    def _normalize_bboxes(
        self, image_shape: Tuple[image_height, image_width], bboxes: np.ndarray
    ) -> np.ndarray:
        if len(bboxes) > 0:
            h, w = image_shape

            scaled_bboxes = bboxes.copy()

            scaled_bboxes = scaled_bboxes.astype(np.float32)

            scaled_bboxes[:, 0] /= w
            scaled_bboxes[:, 1] /= h
            scaled_bboxes[:, 2] /= w
            scaled_bboxes[:, 3] /= h
        else:
            scaled_bboxes = bboxes

        return scaled_bboxes

    @staticmethod
    def _xywh2xyxy(bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) > 0:
            converted_bboxes = bboxes.copy()

            converted_bboxes[:, 2] += converted_bboxes[:, 0]
            converted_bboxes[:, 3] += converted_bboxes[:, 1]

        else:
            converted_bboxes = bboxes

        return converted_bboxes

    @staticmethod
    def _cxcywh2xyxy(bboxes: np.ndarray) -> np.ndarray:
        converted_bboxes = np.copy(bboxes)
        converted_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
        converted_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
        converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # bottom right x
        converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # bottom right y

        return converted_bboxes

    def draw_predictions(
        self,
        image: np.array,
        results: dict,
        colors: dict = None,
        transparency: float = 0,
    ) -> np.array:
        annotation_image = image.copy()
        height, width, _ = image.shape

        if results["labels"] is not None:
            labels = results["labels"]
        else:
            labels = results["categories"]

        for i, (bbox, label, score) in enumerate(
            zip(results["bboxes"], labels, results["confidences"])
        ):
            dimensions = results.get("object_widths")
            positions = results.get("positions")
            distances = results.get("distances")
            if distances is not None:
                np_distances = np.array(distances[i])
                median_distance = np.median(np_distances[np_distances > 0])

            dimension = dimensions[i] if dimensions is not None else None
            distance = median_distance if distances is not None else None
            position = positions[i] if positions is not None else None

            color = self.get_color(colors=colors, label=label)

            x1, y1, x2, y2 = self.scale_bbox_coordinates(bbox, width, height)

            cv2.rectangle(annotation_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            self._display_label(
                label,
                score,
                dimension,
                distance,
                position,
                x1,
                y1,
                x2,
                y2,
                annotation_image,
                color,
            )

        annotation_image = cv2.addWeighted(
            annotation_image, 1 - transparency, image, transparency, 0
        )
        inference_time = results["inference_time"]

        annotation_image = cv2.putText(
            annotation_image,
            f"FPS: {int(1/inference_time)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return annotation_image

    @staticmethod
    def get_color(colors: dict = None, label: str = None) -> tuple:
        if colors is None:
            color = (255, 0, 0)
        else:
            color = colors[label]
            color = (int(color[0]), int(color[1]), int(color[2]))

        return color

    @staticmethod
    def scale_bbox_coordinates(bbox, width, height):
        x1, y1, x2, y2 = bbox

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        return x1, y1, x2, y2

    def _display_label(
        self,
        label: str,
        score: float,
        dimension: float,
        distance: float,
        position: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        annotation_image: np.array,
        color: tuple,
    ) -> None:
        text = f"{label} {score*100:.2f} %"

        if dimension is not None:
            text += f" - Width: {dimension:.2f} m"

        labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
        # print('labelSize>>',labelSize)
        _x1 = max(0, x1)
        _y1 = max(y1 - labelSize[0][1], 0)
        _x2 = _x1 + labelSize[0][0]
        _y2 = _y1 + labelSize[0][1]
        cv2.rectangle(annotation_image, (_x1, _y1), (_x2, _y2), color, cv2.FILLED)
        cv2.putText(
            annotation_image,
            text,
            (_x1, _y2),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 0),
            1,
        )

        if distance is not None:
            text = f"{distance:.2f} m"

            if position is not None:
                text += f", {position:.1f} deg"

            labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            _x1 = max(0, x1)
            _y1 = max(y2 - labelSize[0][1], 0)
            _x2 = _x1 + labelSize[0][0]
            _y2 = _y1 + labelSize[0][1]
            cv2.rectangle(annotation_image, (_x1, _y1), (_x2, _y2), color, cv2.FILLED)
            cv2.putText(
                annotation_image,
                text,
                (_x1, _y2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 0),
                1,
            )
