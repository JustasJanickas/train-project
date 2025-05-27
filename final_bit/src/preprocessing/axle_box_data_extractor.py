import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from preprocessing.frame_transformation import FrameTransformation
from preprocessing.labeling import Labeling
from utils.coco_object import CocoObject
from utils.file_writer import FileWriter
from utils.polygon import Polygon
from utils.video_iterator import VideoIterator


class OnConflict(str, Enum):
    ABORT = "abort"
    APPEND = "append"
    REPLACE = "replace"


class AxleBoxDataExtractor:
    def __init__(self, video_iterator: VideoIterator, output_dir_path: Path) -> None:
        self._iterator = video_iterator

        self._output_dir_path = output_dir_path
        self._positive_dir_path = self._output_dir_path / "positive"
        self._negative_dir_path = self._output_dir_path / "negative"

        self.create_dirs()

    def create_dirs(self):
        self._output_dir_path.mkdir(parents=True, exist_ok=True)
        self._positive_dir_path.mkdir(exist_ok=True)
        self._negative_dir_path.mkdir(exist_ok=True)

    def label_images(
        self,
        transformations: List[FrameTransformation],
        labeling: Labeling,
        coco_objects: Dict[int, CocoObject],
        on_conflict: OnConflict = OnConflict.ABORT,
    ) -> Dict[str, CocoObject]:
        if on_conflict == OnConflict.ABORT and (
            any(os.scandir(self._positive_dir_path))
            or any(os.scandir(self._negative_dir_path))
        ):
            return

        if on_conflict == OnConflict.REPLACE:
            shutil.rmtree(self._output_dir_path)
            self.create_dirs()

        result_dict = {}
        for idx, frame in enumerate(self._iterator):
            if labeling.should_interrupt(idx):
                break

            if (label := labeling.get(idx)) is None:
                continue

            dir_path = self._positive_dir_path if label else self._negative_dir_path
            file_writer = FileWriter(dir_path=dir_path)
            for transformation in transformations:
                transformed_image, transformed_coco_object = transformation.apply_on(
                    frame, coco_objects[idx]
                )

                if any(
                    [polygon.vertices for polygon in transformed_coco_object.polygons]
                ):

                    saved_filepath = file_writer.save_image(
                        image=Image.fromarray(transformed_image), idx=idx
                    )

                    result_dict[saved_filepath.stem] = transformed_coco_object

                    # fig, ax = plt.subplots()
                    # ax.imshow(transformed_image)
                    #
                    # for poly in transformed_coco_object.polygons:
                    #     patch = MplPolygon(
                    #         poly.vertices,
                    #         closed=True,
                    #         edgecolor="red",
                    #         facecolor="none",
                    #         linewidth=2,
                    #     )
                    #     ax.add_patch(patch)
                    #
                    # polygon = Polygon.from_bbox(transformed_coco_object.bbox)
                    # patch = MplPolygon(
                    #     polygon.vertices,
                    #     closed=True,
                    #     edgecolor="blue",
                    #     facecolor="none",
                    #     linewidth=2,
                    # )
                    # ax.add_patch(patch)
                    #
                    # plt.show()
        return result_dict