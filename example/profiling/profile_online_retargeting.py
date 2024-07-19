import time
from pathlib import Path
from typing import List

import numpy as np

from dex_retargeting.constants import (
    get_default_config_path,
    RetargetingType,
    HandType,
    ROBOT_NAMES,
    ROBOT_NAME_MAP,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


def profile_retargeting(retargeting: SeqRetargeting, data: List[np.ndarray]):
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices

    total_time = 0
    for i, joint_pos in enumerate(data):
        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        tic = time.perf_counter()
        qpos = retargeting.retarget(ref_value)
        tac = time.perf_counter()
        total_time += tac - tic

    return total_time


def main():

    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    # Load data
    joint_data = np.load("human_joint_right.pkl", allow_pickle=True)
    data_len = len(joint_data)

    # Vector retargeting
    print(f"Being retargeting profiling with a trajectory of {data_len} hand poses.")
    for robot_name in ROBOT_NAMES:
        config_path = get_default_config_path(
            robot_name,
            RetargetingType.vector,
            HandType.right,
        )
        retargeting = RetargetingConfig.load_from_file(config_path).build()
        total_time = profile_retargeting(retargeting, joint_data)
        print(
            f"Vector retargeting of {ROBOT_NAME_MAP[robot_name]} take {total_time}s in total, fps: {data_len/total_time}hz "
        )

    # DexPilot retargeting
    for robot_name in ROBOT_NAMES:
        if "gripper" in ROBOT_NAME_MAP[robot_name]:
            print(f"Skip {ROBOT_NAME_MAP[robot_name]} for DexPilot retargeting.")
            continue
        config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, HandType.right)
        retargeting = RetargetingConfig.load_from_file(config_path).build()
        total_time = profile_retargeting(retargeting, joint_data)
        print(
            f"DexPilot retargeting of {ROBOT_NAME_MAP[robot_name]} take {total_time}s in total, fps: {data_len/total_time}hz "
        )


if __name__ == "__main__":
    main()
