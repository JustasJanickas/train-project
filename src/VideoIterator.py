import cv2

class VideoIterator:
    def __init__(self, path):
        self._path = path
        self._cap = cv2.VideoCapture(self._path)
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self):
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