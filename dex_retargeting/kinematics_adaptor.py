from abc import abstractmethod
from typing import List

import numpy as np

from dex_retargeting.robot_wrapper import RobotWrapper


class KinematicAdaptor:
    def __init__(self, robot: RobotWrapper, target_joint_names: List[str]):
        self.robot = robot
        self.target_joint_names = target_joint_names

        # Index mapping
        self.pin2target = np.array([robot.get_joint_index(n) for n in target_joint_names])

    @abstractmethod
    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Adapt the joint position for different kinematics applications.
        Note that the joint order of this qpos is consistent with pinocchio
        Args:
            qpos: the pinocchio qpos

        Returns: the adapted qpos with the same shape as input

        """
        pass

    @abstractmethod
    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Adapt the jacobian for different kinematics applications.
        Note that the joint order of this Jacobian is consistent with pinocchio
        Args:
            jacobian: the original jacobian

        Returns: the adapted jacobian with the same shape as input

        """
        pass


class MimicJointKinematicAdaptor(KinematicAdaptor):
    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        source_joint_names: List[str],
        mimic_joint_names: List[str],
        multipliers: List[float],
        offsets: List[float],
    ):
        super().__init__(robot, target_joint_names)

        self.multipliers = np.array(multipliers)
        self.offsets = np.array(offsets)

        # Joint name check
        union_set = set(mimic_joint_names).intersection(set(target_joint_names))
        if len(union_set) > 0:
            raise ValueError(
                f"Mimic joint should not be one of the target joints.\n"
                f"Mimic joints: {mimic_joint_names}.\n"
                f"Target joints: {target_joint_names}\n"
                f"You need to specify the target joint names explicitly in your retargeting config"
                f" for robot with mimic joint constraints: {target_joint_names}"
            )

        # Indices in the pinocchio
        self.source_joint_idx_in_pin = np.array([robot.get_joint_index(name) for name in source_joint_names])
        self.mimic_joint_idx_in_pin = np.array([robot.get_joint_index(name) for name in mimic_joint_names])

        # Indices in the output results
        self.source_joint_idx_in_target = np.array([self.target_joint_names.index(n) for n in source_joint_names])

        # Dimension check
        len_source, len_mimic = self.source_joint_idx_in_target.shape[0], self.mimic_joint_idx_in_pin.shape[0]
        len_mul, len_offset = self.multipliers.shape[0], self.offsets.shape[0]
        if not (len_mimic == len_source == len_mul == len_offset):
            raise ValueError(
                f"Mimic joints setting dimension mismatch.\n"
                f"Source joints: {len_source}, mimic joints: {len_mimic}, multiplier: {len_mul}, offset: {len_offset}"
            )
        self.num_active_joints = len(robot.dof_joint_names) - len_mimic

        # Uniqueness check
        if len(mimic_joint_names) != len(np.unique(mimic_joint_names)):
            raise ValueError(f"Redundant mimic joint names: {mimic_joint_names}")

    def forward_qpos(self, pin_qpos: np.ndarray) -> np.ndarray:
        mimic_qpos = pin_qpos[self.source_joint_idx_in_pin] * self.multipliers + self.offsets
        pin_qpos[self.mimic_joint_idx_in_pin] = mimic_qpos
        return pin_qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        target_jacobian = jacobian[..., self.pin2target]
        mimic_joint_jacobian = jacobian[..., self.mimic_joint_idx_in_pin] * self.multipliers
        source_joint_jacobian = np.bincount(
            mimic_joint_jacobian, self.source_joint_idx_in_target, minlength=len(self.target_joint_names)
        )
        return target_jacobian + source_joint_jacobian
