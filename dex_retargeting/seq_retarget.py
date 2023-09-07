import numpy as np
from time import time

from dex_retargeting.optimizer import Optimizer
from dex_retargeting.optimizer_utils import LPFilter
from typing import Optional


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

        # TODO: hack here
        self.scene = None

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
