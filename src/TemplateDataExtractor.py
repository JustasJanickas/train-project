import os
from pathlib import Path
import cv2
from PIL import Image

POSITIVE_LABELS = [
    # [i for i in range(10)],
    # [i for i in range(100, 200)],
]

POSITIVE_LABELS = [item for sublist in POSITIVE_LABELS for item in sublist]


class TemplateDataExtractor:
    def __init__(self, video_iterator, dir_path):
        self._iterator = video_iterator

        self._dir_path = dir_path
        if not self._dir_path.endswith("/"):
            self._dir_path += "/"

        self._positive_dir = self._dir_path + "positive/"
        self._negative_dir = self._dir_path + "negative/"

        Path(self._positive_dir).mkdir(parents=True, exist_ok=True)
        Path(self._negative_dir).mkdir(parents=True, exist_ok=True)

    def create_training_images(self):
        if any(os.scandir(self._positive_dir)) or any(os.scandir(self._negative_dir)):
            return

        for idx, frame in enumerate(self._iterator):
            if idx % 20:
                continue

            height = frame.shape[0]
            width = frame.shape[1]
            square_length = height // 2

            frame_central_lower_square = frame[square_length:2*square_length, (width - square_length) // 2: (width + square_length) // 2, :]

            resized_frame_square = cv2.resize(
                frame_central_lower_square,
                None,
                fx = 227 / square_length,
                fy = 227 / square_length,
                interpolation = cv2.INTER_LINEAR
            )

            dir = self._positive_dir if idx in POSITIVE_LABELS else self._negative_dir
            Image.fromarray(resized_frame_square).save(f"{dir}{idx}.png")
