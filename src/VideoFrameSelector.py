from collections import namedtuple

import cv2


SimilarityMatch = namedtuple("SimilarityMatch", ["score", "top_left", "height", "width"])

class VideoFrameSelector:
    def __init__(self, video_iterator, template_manager):
        self._iterator = video_iterator
        self._template_manager = template_manager

    def get_best_frames(self):
        matches = self.get_list_of_matches()
        moving_average = self.moving_average([match.score for match in matches])

    def get_list_of_matches(self):
        resize_factor = 10
        templates = self._template_manager.load_templates()

        matches = []
        for frame in self._iterator:
            height, width, _ = frame.shape
            # Select only one sixth part of the image (lower middle) and scale down 10 times for easier matching
            resized_frame = cv2.resize(
                frame[height // 2:, width // 3: 2 * width // 3, :],
                None,
                fx = 1 / resize_factor,
                fy = 1 / resize_factor,
                interpolation = cv2.INTER_LINEAR
            )

            best_match = SimilarityMatch(0, 0, 0, 0)
            for template in templates:
                resized_template = cv2.resize(
                    template,
                    None,
                    fx = 1 / resize_factor,
                    fy = 1 / resize_factor,
                    interpolation=cv2.INTER_LINEAR
                )
                match = self.measure_similarity(resized_frame, resized_template)

                if match.score > best_match.score:
                    best_match = match

            # Since similarity score is calculated on the resized and cropped frame, need to adjust
            # the best match coordinates to refer to the original image
            matches.append(
                SimilarityMatch(
                    best_match.score,
                    (
                        (best_match.top_left[0] + resized_frame.shape[1]) * resize_factor,
                        (best_match.top_left[1] + resized_frame.shape[0]) * resize_factor
                    ),
                    best_match.height * resize_factor,
                    best_match.width * resize_factor
                )
            )

        return matches

    @staticmethod
    def moving_average(values, average_range = 50):
        moving_average = []
        for i in range(len(values)):
            half_range = min(average_range // 2, i, len(values) - i - 1)

            sublist = [values[j] for j in range(max(i - half_range, 0), min(i + half_range + 1, len(values)))]
            moving_average.append(sum(sublist) / len(sublist))

        return moving_average

    @staticmethod
    def measure_similarity(frame, template):
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        height, width, _ = template.shape

        return SimilarityMatch(max_val, top_left, height, width)
