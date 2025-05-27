from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
from pydantic import BaseModel

from utils.coco_object import CocoObject
from utils.polygon import Polygon


class FrameTransformation(BaseModel, ABC):
    @abstractmethod
    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        pass


class Identity(FrameTransformation):
    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        return frame, coco_object


class SquareAt(FrameTransformation):
    transformation: FrameTransformation = Identity()
    corner_x: int
    corner_y: int
    size: int

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        frame = frame[
            self.corner_y : self.corner_y + self.size,
            self.corner_x : self.corner_x + self.size,
            :,
        ]

        polygons = [
            polygon.add((-self.corner_x, -self.corner_y)).validate_vertices(
                self.size, self.size
            )
            for polygon in coco_object.polygons
        ]
        coco_object = CocoObject(
            polygons=polygons,
            bbox=Polygon(
                vertices=[
                    vertice for polygon in polygons for vertice in polygon.vertices
                ]
            ).to_bbox(),
            area=coco_object.area,
        )

        return frame, coco_object


class RescaleTo(FrameTransformation):
    transformation: FrameTransformation = Identity()
    scale_x: int
    scale_y: int

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        height, width, _ = frame.shape
        frame = cv2.resize(
            frame,
            None,
            fx=self.scale_x / width,
            fy=self.scale_y / height,
            interpolation=cv2.INTER_LINEAR,
        )

        polygons = [
            polygon.resize(
                scale_x=self.scale_x / width,
                scale_y=self.scale_y / height,
            ).validate_vertices(self.scale_x, self.scale_y)
            for polygon in coco_object.polygons
        ]
        coco_object = CocoObject(
            polygons=polygons,
            bbox=Polygon(
                vertices=[
                    vertice for polygon in polygons for vertice in polygon.vertices
                ]
            ).to_bbox(),
            area=coco_object.area,
        )

        return frame, coco_object


class LowerHalfCentralSquare(FrameTransformation):
    transformation: FrameTransformation = Identity()

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        height, width, _ = frame.shape
        square_length = height // 2
        corner_x = (width - square_length) // 2
        corner_y = height - square_length

        frame, coco_object = SquareAt(
            corner_x=corner_x, corner_y=corner_y, size=square_length
        ).apply_on(frame, coco_object)
        frame, coco_object = RescaleTo(scale_x=227, scale_y=227).apply_on(
            frame, coco_object
        )

        return frame, coco_object


class LowerThreeFifthsCentralSquare(FrameTransformation):
    transformation: FrameTransformation = Identity()

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        height, width, _ = frame.shape
        square_length = (3 * height) // 5
        corner_x = (width - square_length) // 2
        corner_y = height - square_length

        return RescaleTo(
            transformation=SquareAt(
                corner_x=corner_x, corner_y=corner_y, size=square_length
            ),
            scale_x=227,
            scale_y=227,
        ).apply_on(frame, coco_object)


class BrakeRegionSquare(FrameTransformation):
    transformation: FrameTransformation = Identity()

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        square_length = 640
        corner_x = 2420
        corner_y = 1210

        return SquareAt(
            corner_x=corner_x, corner_y=corner_y, size=square_length
        ).apply_on(frame, coco_object)


class Rotation(FrameTransformation):
    transformation: FrameTransformation = Identity()
    angle: int

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        height, width, _ = frame.shape
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        frame = cv2.warpAffine(
            frame,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        polygons = [
            polygon.frontal_multiply(rotation_matrix).validate_vertices(width, height)
            for polygon in coco_object.polygons
        ]
        coco_object = CocoObject(
            polygons=polygons,
            bbox=Polygon(
                vertices=[
                    vertice for polygon in polygons for vertice in polygon.vertices
                ]
            ).to_bbox(),
            area=coco_object.area,
        )

        return frame, coco_object


class HorizontalInverse(FrameTransformation):
    transformation: FrameTransformation = Identity()

    def apply_on(
        self, frame: np.ndarray, coco_object: CocoObject
    ) -> Tuple[np.ndarray, CocoObject]:
        frame, coco_object = self.transformation.apply_on(frame, coco_object)

        height, width, _ = frame.shape
        frame = frame[:, ::-1, :]

        polygons = [
            polygon.negate_x().add((width, 0)).validate_vertices(width, height)
            for polygon in coco_object.polygons
        ]
        coco_object = CocoObject(
            polygons=polygons,
            bbox=Polygon(
                vertices=[
                    vertice for polygon in polygons for vertice in polygon.vertices
                ]
            ).to_bbox(),
            area=coco_object.area,
        )

        return frame, coco_object
