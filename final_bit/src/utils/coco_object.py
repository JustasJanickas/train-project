from typing import List, Tuple

from pydantic import BaseModel

from utils.polygon import Polygon


class CocoObject(BaseModel):
    polygons: List[Polygon]
    bbox: Tuple[float, float, float, float]
    area: float

    @classmethod
    def from_annotation(cls, coco_annotation: dict) -> "CocoObject":
        polygons = []
        segmentations = coco_annotation.get("segmentation", [])

        for segmentation in segmentations:
            if len(segmentation) % 2 != 0:
                raise ValueError(
                    "Segmentation list must contain an even number of coordinates."
                )
            vertices = [
                (segmentation[i], segmentation[i + 1])
                for i in range(0, len(segmentation), 2)
            ]
            polygons.append(Polygon(vertices=vertices))

        bbox = coco_annotation["bbox"]
        area = coco_annotation["area"]

        return cls(polygons=polygons, bbox=bbox, area=area)

    def simple_hash(self, s) -> int:
        hash_value = 0
        for char in s:
            hash_value = (hash_value * 31 + ord(char)) % (2 ** 32)
        return hash_value

    def as_image_description_json_string(self, name) -> str:
        return f"""\
{{
  "id": {self.simple_hash(name)},
  "width": 640,
  "height": 640,
  "file_name": "{name}.png",
  "license": 0,
  "date_captured": ""
}}\
"""

    def as_mask_description_json_string(self, name) -> str:
        if not hasattr(CocoObject.as_mask_description_json_string, "counter"):
            CocoObject.as_mask_description_json_string.counter = 0  # initialize once
        CocoObject.as_mask_description_json_string.counter += 1

        return f"""\
{{
  "segmentation": [
    {",\n".join([f"[{",".join([str(vertex) for vertex in vertices])}]" for vertices in [[item for nested_tuple in polygon.vertices for item in nested_tuple] for polygon in self.polygons]])}
  ],
  "area": {self.area},
  "bbox": [{",".join([str(val) for val in self.bbox])}],
  "iscrowd": 0,
  "id": {CocoObject.as_mask_description_json_string.counter},
  "image_id": {self.simple_hash(name)},
  "category_id": 0
}}\
"""
