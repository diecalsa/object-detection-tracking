import cv2
import numpy as np


class MultiTracker:
    def __init__(
        self, bboxes: list, frame: np.array, tracker_type: str = "csrt"
    ) -> None:
        self.tracker_type = tracker_type
        self.multiTracker = cv2.legacy.MultiTracker_create()
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.legacy.TrackerCSRT_create,
            "kcf": cv2.legacy.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.legacy.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create,
        }

        self.add(bboxes=bboxes, frame=frame)

    def add(self, bboxes: list, frame: np.array) -> None:
        """Adds the given bboxes to the MultiTracker.

        Args:
            bboxes (list): Bboxes to track.
            frame (np.array): Reference frame.
        """

        for bbox in bboxes:
            bbox = np.array(bbox)
            if bbox.dtype == np.float32 or bbox.dtype == np.float64:
                bbox = self.__scale_bbox(bbox, frame)
            new_tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]()
            self.multiTracker.add(new_tracker, frame, tuple(bbox))

    def update(self, frame: np.array) -> list:
        """Updates the MultiTracker.

        Args:
            frame (np.array): New frame.

        Returns:
            list: Updated bboxes.
        """
        ret, bboxes = self.multiTracker.update(frame)

        normalized_bboxes_list = self.__normalize_bbox_coordinates(bboxes, frame)

        return normalized_bboxes_list

    @staticmethod
    def __normalize_bbox_coordinates(bboxes: np.array, frame: np.array) -> list:
        """Scales the given bbox from [[0-w], [0-h]] to [0-1].

        Args:
            bboxes (np.array): [description]
            frame (np.array): [description]

        Returns:
            list: [description]
        """
        h, w, _ = frame.shape

        if len(bboxes) > 0:
            normalized_bboxes = bboxes.copy()
            normalized_bboxes[:, 0] /= w
            normalized_bboxes[:, 1] /= h
            normalized_bboxes[:, 2] = (bboxes[:, 0] + bboxes[:, 2]) / w
            normalized_bboxes[:, 3] = (bboxes[:, 1] + bboxes[:, 3]) / h

            normalized_bboxes_list = normalized_bboxes.tolist()
        else:
            normalized_bboxes_list = []

        return normalized_bboxes_list

    @staticmethod
    def __scale_bbox(bbox: np.array, frame: np.array) -> np.array:
        """Scales the given bbox from [0-1] to [[0-w], [0-h]].

        Args:
            bbox (np.array): Given bbox [float]
            frame (np.array): Current frame to analyse.

        Returns:
            np.array: Scaled bbox [int]
        """
        h, w, _ = frame.shape
        scaled_bbox = (
            int(bbox[0] * w),
            int(bbox[1] * h),
            int((bbox[2] - bbox[0]) * w),
            int((bbox[3] - bbox[1]) * h),
        )

        return scaled_bbox
