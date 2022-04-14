from .darknet_model import DarknetModel
import cv2
import numpy as np

import typing
from typing import Tuple

bboxes = scores = categories = np.ndarray
image_height = image_width = int


class YoloV4Model(DarknetModel):
    def __init__(
        self,
        weights_file: str,
        cfg_file: str,
        image_size: Tuple[int, int] = (416, 416),
    ):
        super().__init__(
            weights_file=weights_file, cfg_file=cfg_file, image_size=image_size
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def normalize_predictions(
        self, image_shape: Tuple[image_height, image_width], predictions: np.ndarray
    ) -> Tuple[bboxes, scores, categories]:

        bboxes, scores, categories = predictions
        xyxy_bboxes = self._xywh2xyxy(bboxes)
        scaled_bboxes = self._normalize_bboxes(image_shape, xyxy_bboxes)

        return np.array(scaled_bboxes), np.array(scores), np.array(categories)
