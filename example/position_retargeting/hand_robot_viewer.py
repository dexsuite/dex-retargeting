from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
import transforms3d.quaternions

from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from hand_viewer import HandDatasetSAPIENViewer

ROBOT2MANO = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
ROBOT2MANO_POSE = sapien.Pose(q=transforms3d.quaternions.mat2quat(ROBOT2MANO))


def prepare_position_retargeting(joint_pos: np.array, link_hand_indices: np.ndarray):
    link_pos = joint_pos[link_hand_indices]
    return link_pos


def prepare_vector_retargeting(joint_pos: np.array, link_hand_indices_pairs: np.ndarray):
    joint_pos = joint_pos @ ROBOT2MANO
    origin_link_pos = joint_pos[link_hand_indices_pairs[0]]
    task_link_pos = joint_pos[link_hand_indices_pairs[1]]
    return task_link_pos - origin_link_pos


class LPFilter:
    def __init__(self, control_freq, cutoff_freq):
        dt = 1 / control_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos**2 + 2 * y_cos)
        self.y = 0
        self.is_init = False

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def init(self, y):
        self.y = y.copy()
        self.is_init = True


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(self, robot_names: List[RobotName], hand_type: HandType, headless=False, use_ray_tracing=False):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []

        # Load optimizer and filter
        for robot_name in robot_names:
            config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build(self.scene)
            robot = retargeting.optimizer.robot
            robot_file_name = Path(config.urdf_path).stem
            self.robots.append(robot)
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset, 0]))
        if self.viewer is not None:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        # Loop rendering
        for i in range(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(pose_offsets[copy_ind] * pose)

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting in zip(self.robots, self.retargetings):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)
                robot.set_qpos(qpos)

            for k in range(60 // fps):
                self.viewer.render()

        self.viewer.toggle_pause(True)
        self.viewer.render()
