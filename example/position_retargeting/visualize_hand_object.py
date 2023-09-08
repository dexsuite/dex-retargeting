import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer
from dex_retargeting.retargeting_config import RetargetingConfig


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int):
    dataset = DexYCBVideoDataset(data_root)
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=False)
    else:
        viewer = RobotHandDatasetSAPIENViewer(list(robots), HandType.right, headless=False)

    sampled_data = dataset[4]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps)


def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, fps: int = 5):
    print(robots)
    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    viz_hand_object(robots, data_root, fps)


if __name__ == "__main__":
    tyro.cli(main)
