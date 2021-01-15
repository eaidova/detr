from pathlib import Path
from .coco import CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided VG path {root} does not exist'
    PATHS = {
        "train": (root, f'{root}/coco_format/train.json'),
        "val": (root, f'{root}/coco_format/val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)

    return dataset
