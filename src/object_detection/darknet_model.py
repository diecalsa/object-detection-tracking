import cv2
import numpy as np

from typing import Tuple

from .abstract_detection_model import AbstractDetectionModel

bboxes = scores = categories = np.ndarray


class DarknetModel(AbstractDetectionModel):
    def __init__(
        self,
        weights_file: str,
        cfg_file: str,
        image_size: Tuple[int, int] = (416, 416),
    ) -> None:
        super().__init__(weights_file=weights_file, image_size=image_size)
        self._cfg_file = cfg_file

    def load(self) -> None:
        net = cv2.dnn.readNet(self._weights_file, self._cfg_file)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=self._image_size, scale=1 / 255, swapRB=False)

    def predict(self, image: np.ndarray) -> Tuple[bboxes, scores, categories]:
        categories, scores, xywh_bboxes = self.model.detect(image, 0.1, 0.1)

        return xywh_bboxes, scores, categories
