import numpy as np
import time
import os
import cv2

from typing import Tuple
import logging

from .abstract_detection_model import AbstractDetectionModel

logger = logging.getLogger(__name__)


class ObjectDetectionModel:
    """Context class. Interface with User."""

    def __init__(
        self,
        model: AbstractDetectionModel,
        labels_file: str = None,
        min_score: float = 0.3,
        nms_score: float = 0.5,
    ) -> None:

        self._model = model
        self._model.load()

        self._labelmap = (
            self.load_labelmap(labels_file) if labels_file is not None else None
        )
        self.min_score = min_score
        self.nms_score = nms_score
        self._colors = self.create_colormap(self._labelmap)

    def predict(self, image: np.ndarray, min_score_by_class: dict = None) -> dict:
        """[summary]

        :param image: [description]
        :type image: np.ndarray
        :param min_score_by_class: [description], defaults to None
        :type min_score_by_class: dict, optional
        :return: [description]
        :rtype: dict
        """

        image_height, image_width, _ = image.shape
        image = self._model.preprocess(image)

        start = time.time()
        predictions = self._model.predict(image)
        bboxes, scores, categories = self._model.normalize_predictions(
            (image_height, image_width), predictions
        )
        end = time.time()

        if bboxes.shape[0] and scores.shape[0] and categories.shape[0]:
            bboxes, scores, categories = self.postprocess(bboxes, scores, categories)
        else:
            bboxes, scores, categories = [], [], []
        if self._labelmap is not None:
            labels = [self._labelmap[cat] for cat in categories]
        else:
            labels = None
        inference_time = end - start

        results = {
            "bboxes": bboxes,
            "confidences": scores,
            "categories": categories,
            "labels": labels,
            "inference_time": inference_time,
        }

        return results

    def postprocess(
        self, bboxes: np.ndarray, scores: np.ndarray, categories: np.ndarray
    ) -> Tuple[list, list, list]:

        nms_bboxes = list(bboxes)
        nms_scores = np.array(scores)

        indices = cv2.dnn.NMSBoxes(
            nms_bboxes, nms_scores, self.min_score, self.nms_score
        )

        intermediate_bboxes = []
        final_bboxes = []
        final_confidences = []
        final_categories = []

        nms_bboxes = np.array(nms_bboxes).reshape((-1, 4))

        for idx in indices:
            intermediate_bboxes.append(nms_bboxes[idx])
            final_confidences.append(float(nms_scores[idx]))
            final_categories.append(int(categories[idx]))

        if len(intermediate_bboxes) > 0:
            for bbox in intermediate_bboxes:
                x1, y1, x2, y2 = bbox.reshape((-1))

                bbox = [x1, y1, x2, y2]
                final_bboxes.append(bbox)

        return final_bboxes, final_confidences, final_categories

    def draw_predictions(
        self, image: np.array, results: dict, transparency: float = 0
    ) -> np.array:
        return self._model.draw_predictions(
            image, results, colors=self._colors, transparency=transparency
        )

    @staticmethod
    def load_labelmap(label_map: str) -> dict:
        logger.info("Loading labels file...")
        if not os.path.exists(label_map):
            logger.warning(
                f"Could not load labels file {label_map}. Check if the path exists."
            )
            return None
        with open(label_map, "r") as f:
            content = f.read()
            lines = content.splitlines()

        labelmap = {i: line for i, line in enumerate(lines)}
        logger.debug(f"Labels file loaded: {labelmap}")
        logger.info("Labels file loaded.")
        return labelmap

    @staticmethod
    def create_colormap(labelmap: dict = None) -> dict:
        if labelmap is None:
            return None

        colors = {}
        np.random.seed(0)
        for key, value in labelmap.items():
            colors[value] = tuple(np.random.randint(100, 255, (3)))

        logger.debug(colors)
        return colors
