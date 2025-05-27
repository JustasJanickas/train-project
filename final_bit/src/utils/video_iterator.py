from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class VideoIterator(Iterator[np.ndarray]):
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._count = 0

    def __iter__(self) -> "VideoIterator":
        return self

    def __next__(self) -> np.ndarray:
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))

        if self._count % 100 == 0:
            print(f"Progress: {self._count}")
        self._count += 1

        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # If we cannot read a frame (end of video), stop iteration
                self._cap.release()
                print(f"Progress: {self._count}")
                raise StopIteration
            return frame
        else:
            # If video capture cannot be opened, raise StopIteration
            raise StopIteration

    def reset(self):
        self._cap = None
        self._count = 0
