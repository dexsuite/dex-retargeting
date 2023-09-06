import argparse
import os
from pathlib import Path

from dataset import DexYCBVideoDataset
from dex_toolkit.retargeting.optimizer import PositionOptimizer, VectorOptimizer
from dex_toolkit.retargeting.seq_retarget import SeqRetargeting
from hand_robot_viewer import DatasetHandRobotViewer
from hand_viewer import DatasetHandViewer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mode",
                        default='adroit',
                        choices=['allegro', 'adroit', 'human'])
    return parser.parse_args()


def viz_human_object():
    data_root = os.environ["DEX_YCB_DIR"] if "DEX_YCB_DIR" in os.environ else str(Path.home() / "data" / "dex_ycb")
    dataset = DexYCBVideoDataset(data_root)
    replay = DatasetHandViewer()
    sampled_data = dataset[4]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    replay.load_object_hand(sampled_data)
    replay.render_data_frames(sampled_data, fps=5)


def viz_robot_object():
    # Create Dex-YCB dataset
    data_root = os.environ["DEX_YCB_DIR"] if "DEX_YCB_DIR" in os.environ else str(Path.home() / "data" / "dex_ycb")
    dataset = DexYCBVideoDataset(data_root)
    sampled_data = dataset[2]

    # Create simulation env
    robot_name = "allegro_hand_free"
    env = DatasetHandRobotViewer(gui=True, robot_name=robot_name)
    robot = env.robots[0]
    link_names = ["palm", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                  "link_2.0", "link_6.0", "link_10.0"]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    link_hand_indices = [0, 4, 8, 12, 16] + [2, 6, 10, 14]
    optimizer = PositionOptimizer(robot, joint_names, link_names)
    retargeting_optimizer = SeqRetargeting(optimizer, has_joint_limits=True)

    # Loop data
    env.load_object_hand(sampled_data)
    env.render_robot_data_frames(sampled_data, retargeting_optimizer, link_hand_indices, fps=10)

    retargeting_optimizer.verbose()


def viz_robot_object_adroit_hand():
    # Create Dex-YCB dataset
    data_root = os.environ["DEX_YCB_DIR"] if "DEX_YCB_DIR" in os.environ else str(Path.home() / "data" / "dex_ycb")
    dataset = DexYCBVideoDataset(data_root)
    sampled_data = dataset[2]

    # Create simulation env
    robot_name = "adroit_hand_free"
    env = DatasetHandRobotViewer(gui=True, robot_name=robot_name)
    link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"] + ["thmiddle", "ffmiddle", "mfmiddle",
                                                                          "rfmiddle", "lfmiddle"]
    robot = env.robots[0]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    link_hand_indices = [0, 4, 8, 12, 16, 20] + [2, 6, 10, 14, 18]
    optimizer = PositionOptimizer(robot, joint_names, link_names)
    retargeting_optimizer = SeqRetargeting(optimizer, has_joint_limits=True)

    # Loop data
    env.load_object_hand(sampled_data)
    env.render_robot_data_frames(sampled_data, retargeting_optimizer, link_hand_indices, fps=10)

    retargeting_optimizer.verbose()


def viz_robot_object_vec():
    # Create Dex-YCB dataset
    data_root = os.environ["DEX_YCB_DIR"] if "DEX_YCB_DIR" in os.environ else str(Path.home() / "data" / "dex_ycb")
    dataset = DexYCBVideoDataset(data_root)
    sampled_data = dataset[2]

    # Create simulation env
    robot_name = "allegro_hand"
    env = DatasetHandRobotViewer(gui=True, robot_name=robot_name)
    robot = env.robots[0]
    origin_link_names = ["wrist", "wrist", "wrist", "wrist"]
    task_link_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    link_hand_indices = [[0, 0, 0, 0], [4, 8, 12, 16]]
    optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                task_link_names=task_link_names, scaling=1.6)
    retargeting_optimizer = SeqRetargeting(optimizer, has_joint_limits=True)

    # Loop data
    env.load_object_hand(sampled_data)
    env.render_robot_data_frames(sampled_data, retargeting_optimizer, link_hand_indices, fps=10)

    retargeting_optimizer.verbose()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == "human":
        viz_human_object()
    elif args.mode == "allegro":
        viz_robot_object_vec()
    elif args.mode == "adroit":
        viz_robot_object_adroit_hand()
    else:
        raise ValueError(f"Unknown mode {args.mode}")
