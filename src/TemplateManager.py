import os
import glob

import numpy as np
from PIL import Image

from src.VideoIterator import VideoIterator


class TemplateManager:
    def __init__(self, dir_path):
        self._path = dir_path
        if not self._path.endswith("/"):
            self._path += "/"

    def load_templates(self):
        png_files = glob.glob(os.path.join(self._path, "*.png"))
        templates = []
        for file in png_files:
            img = Image.open(file)
            templates.append(np.array(img))

        return templates

    def populate_templates(self, video_path = "../resources/videos/20241203_104653.mp4"):
        video_iterator = VideoIterator(video_path)

        frame_crops = {
            0: lambda x: x[1420:2070, 2590:3240, :],
            8650: lambda x: x[1405:2055, 1920:2570, :],
        }
        for idx, frame in enumerate(video_iterator):
            # Early exit condition. Do not iterate over whole video if unnecessary
            if idx > max(frame_crops.keys()):
                break

            if idx not in frame_crops.keys():
                continue

            crop_frame = frame_crops[idx]

            Image.fromarray(crop_frame(frame)).save(f"{self._path}box_{idx}.png")

        print(f"Progress: {idx}")
