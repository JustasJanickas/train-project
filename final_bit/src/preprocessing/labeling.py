from typing import List, Optional

from pydantic import BaseModel


class Labeling(BaseModel):
    frame_idxs: List[int]
    positive_labels: List[int]

    def should_interrupt(self, frame_idx: int) -> bool:
        return frame_idx > max(self.frame_idxs)

    def get(self, idx) -> Optional[bool]:
        return None if idx not in self.frame_idxs else idx in self.positive_labels
