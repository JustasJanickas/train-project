import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

from src.TemplateDataExtractor import TemplateDataExtractor

class VideoFrameSelector:
    def __init__(self, video_iterator, ensemble):
        self._iterator = video_iterator
        self._ensemble = ensemble

    def get_best_frames(self, frame_limit=None):
        scores = self.get_frame_scores(frame_limit)
        moving_average = self.moving_average(scores)
        peaks, _ = find_peaks(moving_average, height=0.5, distance=100)
        return peaks, scores

    def get_frame_scores(self, frame_limit=None):
        scores = []
        for idx, frame in enumerate(self._iterator):
            if frame_limit is not None and idx == frame_limit:
                break
            central_square = TemplateDataExtractor.get_resized_central_square(frame)
            tensor_input = tf.convert_to_tensor(np.expand_dims(central_square / 255.0, axis=0), dtype=tf.float32)
            score = self._ensemble.predict_single(tensor_input, verbose=0)
            scores.append(score)

        return scores

    @staticmethod
    def moving_average(values, average_range = 50):
        half_range = average_range // 2
        padded_values = [values[0]] * half_range + values + [values[-1]] * half_range

        moving_average = []
        for i in range(half_range, len(padded_values) - half_range):

            sublist = padded_values[i - half_range: i + half_range]
            moving_average.append(sum(sublist) / len(sublist))

        return moving_average
