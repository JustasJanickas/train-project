import json
import random
from pathlib import Path

from preprocessing.axle_box_data_extractor import (AxleBoxDataExtractor,
                                                   OnConflict)
from preprocessing.frame_transformation import (BrakeRegionSquare,
                                                HorizontalInverse,
                                                Rotation)
from preprocessing.labeling import Labeling
from utils.coco_object import CocoObject
from utils.video_iterator import VideoIterator


def main():
    with open(
        "../resources/coco_files/via_project_22May2025_18h31m_coco.json", "r"
    ) as f:
        coco_data = json.load(f)

    coco_objects = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        coco_objects[image_id] = CocoObject.from_annotation(annotation)


    # Brake data collection
    iterator = VideoIterator(video_path=Path("../resources/videos/20241203_104653.mp4"))
    frames_list = [
        176,
        1011,
        1311,
        1699,
        2016,
        3061,
        3377,
        3762,
        4097,
        5418,
        5863,
        6356,
        6722,
        7831,
        8149,
        8627,
        8887,
        11347,
        11666,
        12398,
        12726,
        15610,
        15918,
        16515,
        16853,
        18281,
        18542,
    ]
    labeling = Labeling(frame_idxs=frames_list, positive_labels=frames_list)
    train_data_extractor = AxleBoxDataExtractor(
        video_iterator=iterator, output_dir_path=Path("../resources/brake_images")
    )

    transformations = [
        BrakeRegionSquare(),
        HorizontalInverse(transformation=BrakeRegionSquare()),
        BrakeRegionSquare(transformation=HorizontalInverse()),
        HorizontalInverse(
            transformation=BrakeRegionSquare(transformation=HorizontalInverse())
        ),
        BrakeRegionSquare(transformation=Rotation(angle=1)),
        HorizontalInverse(
            transformation=BrakeRegionSquare(transformation=Rotation(angle=1))
        ),
        BrakeRegionSquare(
            transformation=HorizontalInverse(transformation=Rotation(angle=1))
        ),
        HorizontalInverse(
            transformation=BrakeRegionSquare(
                transformation=HorizontalInverse(transformation=Rotation(angle=1))
            )
        ),
        BrakeRegionSquare(transformation=Rotation(angle=-1)),
        HorizontalInverse(
            transformation=BrakeRegionSquare(transformation=Rotation(angle=-1))
        ),
        BrakeRegionSquare(
            transformation=HorizontalInverse(transformation=Rotation(angle=-1))
        ),
        HorizontalInverse(
            transformation=BrakeRegionSquare(
                transformation=HorizontalInverse(transformation=Rotation(angle=-1))
            )
        ),
        BrakeRegionSquare(transformation=Rotation(angle=2)),
        HorizontalInverse(
            transformation=BrakeRegionSquare(transformation=Rotation(angle=2))
        ),
        BrakeRegionSquare(
            transformation=HorizontalInverse(transformation=Rotation(angle=2))
        ),
        HorizontalInverse(
            transformation=BrakeRegionSquare(
                transformation=HorizontalInverse(transformation=Rotation(angle=2))
            )
        ),
        BrakeRegionSquare(transformation=Rotation(angle=-2)),
        HorizontalInverse(
            transformation=BrakeRegionSquare(transformation=Rotation(angle=-2))
        ),
        BrakeRegionSquare(
            transformation=HorizontalInverse(transformation=Rotation(angle=-2))
        ),
        HorizontalInverse(
            transformation=BrakeRegionSquare(
                transformation=HorizontalInverse(transformation=Rotation(angle=-2))
            )
        ),
    ]

    iterator.reset()
    coco_dict = train_data_extractor.label_images(
        transformations,
        labeling,
        coco_objects,
        OnConflict.REPLACE,
    )

    random.seed(42)
    shuffled_kvs = list(coco_dict.items())
    random.shuffle(shuffled_kvs)

    image_strings_train = [coco.as_image_description_json_string(name) for idx, (name, coco) in enumerate(shuffled_kvs) if idx < 0.8 * len(shuffled_kvs)]
    mask_strings_train = [coco.as_mask_description_json_string(name) for idx, (name, coco) in enumerate(shuffled_kvs) if idx < 0.8 * len(shuffled_kvs)]
    image_strings_test = [coco.as_image_description_json_string(name) for idx, (name, coco) in enumerate(shuffled_kvs) if idx >= 0.8 * len(shuffled_kvs)]
    mask_strings_test = [coco.as_mask_description_json_string(name) for idx, (name, coco) in enumerate(shuffled_kvs) if idx >= 0.8 * len(shuffled_kvs)]

    string_builder_train = f"""\
{{
  "info": {{
    "year": 2025,
    "version": "1.0",
    "description": "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)",
    "contributor": "",
    "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
    "date_created": "Thu May 22 2025 20:15:44 GMT+0300 (Eastern European Summer Time)"
  }},
  "images": [
    {",".join(image_strings_train)}
  ],
  "annotations": [
    {",".join(mask_strings_train)}
  ],
  "licenses": [
    {{
      "id": 0,
      "name": "Unknown License",
      "url": ""
    }}
  ],
  "categories": [
    {{
      "supercategory": "type",
      "id": 0,
      "name": "Brake Pad (object)"
    }}
  ]
}}\
"""

    string_builder_test = f"""\
    {{
      "info": {{
        "year": 2025,
        "version": "1.0",
        "description": "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)",
        "contributor": "",
        "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
        "date_created": "Thu May 22 2025 20:15:44 GMT+0300 (Eastern European Summer Time)"
      }},
      "images": [
        {",".join(image_strings_test)}
      ],
      "annotations": [
        {",".join(mask_strings_test)}
      ],
      "licenses": [
        {{
          "id": 0,
          "name": "Unknown License",
          "url": ""
        }}
      ],
      "categories": [
        {{
          "supercategory": "type",
          "id": 0,
          "name": "Brake Pad (object)"
        }}
      ]
    }}\
    """

    with open("../resources/coco_files/coco_train.json", "w") as file:
        file.write(string_builder_train)
    with open("../resources/coco_files/coco_test.json", "w") as file:
        file.write(string_builder_test)


if __name__ == "__main__":
    main()
