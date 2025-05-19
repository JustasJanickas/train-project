import os
from pathlib import Path
import cv2
from PIL import Image

POSITIVE_LABELS = [
    [i for i in range(135, 220 + 1)],
    [i for i in range(960, 1065 + 1)],
    [i for i in range(1255, 1365 + 1)],
    [i for i in range(1640, 1760 + 1)],
    [i for i in range(1955, 2075 + 1)],
    [i for i in range(3000, 3120 + 1)],
    [i for i in range(3320, 3435 + 1)],
    [i for i in range(3705, 3825 + 1)],
    [i for i in range(4035, 4160 + 1)],
    [i for i in range(5330, 5510 + 1)],
    [i for i in range(5785, 5940 + 1)],
    [i for i in range(6290, 6425 + 1)],
    [i for i in range(6655, 6785 + 1)],
    [i for i in range(7765, 7890 + 1)],
    [i for i in range(8095, 8205 + 1)],
    [i for i in range(8580, 8675 + 1)],
    [i for i in range(8840, 8930 + 1)],
    [i for i in range(11290, 11400 + 1)],
    [i for i in range(11605, 11730 + 1)],
    [i for i in range(12330, 12460 + 1)],
    [i for i in range(12665, 12780 + 1)],
    [i for i in range(15555, 15665 + 1)],
    [i for i in range(15860, 15975 + 1)],
    [i for i in range(16445, 16580 + 1)],
    [i for i in range(16795, 16910 + 1)],
    [i for i in range(18225, 18335 + 1)],
    [i for i in range(18500, 18585 + 1)],
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
            if idx % 10:
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
