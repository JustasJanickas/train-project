from pathlib import Path

from PIL import Image
from pydantic import BaseModel


class FileWriter(BaseModel):
    dir_path: Path

    def save_image(self, image: Image.Image, idx: int) -> Path:
        self.dir_path.mkdir(parents=True, exist_ok=True)

        base_file = self.dir_path / f"{idx}.png"
        if not base_file.exists():
            file_path = base_file
        else:
            i = 1
            while True:
                file_path = self.dir_path / f"{idx}_{i}.png"
                if not file_path.exists():
                    break
                i += 1

        image.save(file_path)
        return file_path
