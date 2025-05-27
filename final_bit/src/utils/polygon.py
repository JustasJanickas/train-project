from typing import List, Tuple

from pydantic import BaseModel


class Polygon(BaseModel):
    vertices: List[Tuple[float, float]]

    @classmethod
    def from_bbox(cls, bbox) -> "Polygon":
        x, y, w, h = bbox
        vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return cls(vertices=vertices)

    def negate_x(self) -> "Polygon":
        vertices = [(-x, y) for x, y in self.vertices]
        return Polygon(vertices=vertices)

    def negate_y(self) -> "Polygon":
        vertices = [(x, -y) for x, y in self.vertices]
        return Polygon(vertices=vertices)

    def add(self, point: Tuple[float, float]) -> "Polygon":
        vertices = [(x + point[0], y + point[1]) for x, y in self.vertices]
        return Polygon(vertices=vertices)

    def resize(self, scale_x: float, scale_y: float) -> "Polygon":
        vertices = [(x * scale_x, y * scale_y) for x, y in self.vertices]
        return Polygon(vertices=vertices)

    def frontal_multiply(self, M) -> "Polygon":
        vertices = [M @ [x, y, 1.0] for x, y in self.vertices]
        return Polygon(vertices=vertices)

    def validate_vertices(self, x_max, y_max) -> "Polygon":
        if any([x < 0 or x >= x_max or y < 0 or y >= y_max for x, y in self.vertices]):
            return Polygon(vertices=[])
        return self

    def to_bbox(self) -> Tuple[float, float, float, float]:
        if not self.vertices:
            return 0.0, 0.0, 0.0, 0.0

        corner_x = min([x for x, _ in self.vertices])
        corner_y = min([y for _, y in self.vertices])
        width_x = max([x for x, _ in self.vertices]) - corner_x
        width_y = max([y for _, y in self.vertices]) - corner_y

        return corner_x, corner_y, width_x, width_y

    # def __repr__(self):
    #     return f"Polygon({self.vertices})"
