from time import time
from typing import Optional

import numpy as np
import transforms3d

from dex_retargeting.optimizer import Optimizer
from dex_retargeting.optimizer_utils import LPFilter


class SeqRetargeting:
    def __init__(
        self,
        optimizer: Optimizer,
        has_joint_limits=True,
        lp_filter: Optional[LPFilter] = None,
    ):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        # Joint limit
        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.get_qlimits())
        joint_limits[:, 0] = -1e4  # a large value is equivalent to no limit
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.get_qlimits()[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.target_joint_indices])
        self.joint_limits = joint_limits

        # Temporal information
        self.last_qpos = joint_limits.mean(1)[self.optimizer.target_joint_indices]
        self.accumulated_time = 0
        self.num_retargeting = 0

        # Filter
        self.filter = lp_filter

        # Warm started
        self.is_warm_started = False

        # TODO: hack here
        self.scene = None

    def warm_start(self, wrist_pos: np.ndarray, wrist_orientation: np.ndarray, global_rot: np.array):
        # This function can only be used when the first joints of robot are free joints
        if len(wrist_pos) != 3:
            raise ValueError(f"Wrist pos:{wrist_pos} is not a 3-dim vector.")
        if len(wrist_orientation) != 3:
            raise ValueError(f"Wrist orientation:{wrist_orientation} is not a 3-dim vector.")

        if np.linalg.norm(wrist_orientation) < 1e-3:
            mat = np.eye(3)
        else:
            angle = np.linalg.norm(wrist_orientation)
            axis = wrist_orientation / angle
            mat = transforms3d.axangles.axangle2mat(axis, angle)
            print(transforms3d.quaternions.axangle2quat(axis, angle))

        robot = self.optimizer.robot
        operator2mano = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        mat = global_rot.T @ mat @ operator2mano
        target_wrist_pose = np.eye(4)
        target_wrist_pose[:3, :3] = mat
        target_wrist_pose[:3, 3] = wrist_pos

        wrist_link_name = self.optimizer.wrist_link_name
        wrist_link = [link for link in self.optimizer.robot.get_links() if link.get_name() == wrist_link_name][0]
        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        old_qpos = robot.get_qpos()
        new_qpos = old_qpos.copy()
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                new_qpos[num] = 0
        robot.set_qpos(new_qpos)
        root2wrist = (robot.get_pose().inv() * wrist_link.get_pose()).inv().to_transformation_matrix()
        target_root_pose = target_wrist_pose @ root2wrist
        robot.set_qpos(old_qpos)

        euler = transforms3d.euler.mat2euler(target_root_pose[:3, :3], "rxyz")
        pose_vec = np.concatenate([target_root_pose[:3, 3], euler])

        # Find the dummy joints
        name_list = [
            "dummy_x_translation_joint",
            "dummy_y_translation_joint",
            "dummy_z_translation_joint",
            "dummy_x_rotation_joint",
            "dummy_y_rotation_joint",
            "dummy_z_rotation_joint",
        ]
        for num, joint_name in enumerate(self.optimizer.target_joint_names):
            if joint_name in name_list:
                index = name_list.index(joint_name)
                self.last_qpos[num] = pose_vec[index]

        self.is_warm_started = True

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        tic = time()
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            fixed_qpos=fixed_qpos.astype(np.float32),
            last_qpos=self.last_qpos.astype(np.float32),
        )
        self.accumulated_time += time() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos
        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.fixed_joint_indices] = fixed_qpos
        robot_qpos[self.optimizer.target_joint_indices] = qpos
        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def verbose(self):
        min_value = self.optimizer.opt.last_optimum_value()
        print(f"Retargeting {self.num_retargeting} times takes: {self.accumulated_time}s")
        print(f"Last distance: {min_value}")

    def reset(self):
        self.last_qpos = self.joint_limits.mean(1)[self.optimizer.target_joint_indices]
        self.num_retargeting = 0
