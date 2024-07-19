from pathlib import Path
from time import time

import numpy as np
import pytest

from dex_retargeting.constants import ROBOT_NAMES, get_default_config_path, RetargetingType, HandType, RobotName
from dex_retargeting.optimizer import VectorOptimizer, PositionOptimizer, Optimizer
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper


class TestOptimizer:
    np.set_printoptions(precision=4)
    config_dir = Path(__file__).parent.parent / "dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))
    DEXPILOT_ROBOT_NAMES = ROBOT_NAMES.copy()
    DEXPILOT_ROBOT_NAMES.remove(RobotName.ability)

    @staticmethod
    def sample_qpos(optimizer: Optimizer):
        joint_eps = 1e-5
        robot = optimizer.robot
        adaptor = optimizer.adaptor
        joint_limit = robot.joint_limits
        random_qpos = np.random.uniform(joint_limit[:, 0], joint_limit[:, 1])
        if adaptor is not None:
            random_qpos = adaptor.forward_qpos(random_qpos)

        init_qpos = np.clip(
            random_qpos + np.random.randn(robot.dof) * 0.5, joint_limit[:, 0] + joint_eps, joint_limit[:, 1] - joint_eps
        )[optimizer.idx_pin2target]
        return random_qpos, init_qpos

    @staticmethod
    def compute_pin_qpos(optimizer: Optimizer, qpos: np.ndarray, fixed_qpos: np.ndarray):
        adaptor = optimizer.adaptor
        full_qpos = np.zeros(optimizer.robot.model.nq)
        full_qpos[optimizer.idx_pin2target] = qpos
        full_qpos[optimizer.idx_pin2fixed] = fixed_qpos
        if adaptor is not None:
            full_qpos = adaptor.forward_qpos(full_qpos)
        return full_qpos

    @staticmethod
    def generate_vector_retargeting_data_gt(robot: RobotWrapper, optimizer: VectorOptimizer):
        random_pin_qpos, init_qpos = TestOptimizer.sample_qpos(optimizer)
        robot.compute_forward_kinematics(random_pin_qpos)
        random_pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices])
        origin_pos = random_pos[optimizer.origin_link_indices]
        task_pos = random_pos[optimizer.task_link_indices]
        random_target_vector = task_pos - origin_pos

        return random_pin_qpos, init_qpos, random_target_vector

    @staticmethod
    def generate_position_retargeting_data_gt(robot: RobotWrapper, optimizer: PositionOptimizer):
        random_pin_qpos, init_qpos = TestOptimizer.sample_qpos(optimizer)
        robot.compute_forward_kinematics(random_pin_qpos)
        random_target_pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.target_link_indices])

        return random_pin_qpos, init_qpos, random_target_pos

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_position_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)

        # Note: The parameters below are adjusted solely for this test
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert isinstance(retargeting.optimizer, PositionOptimizer)

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random position
            random_qpos, init_qpos, random_target_pos = self.generate_position_retargeting_data_gt(robot, optimizer)
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Optimized position
            computed_qpos = optimizer.retarget(random_target_pos, fixed_qpos=fixed_qpos, last_qpos=init_qpos[:])

            # Check results
            robot.compute_forward_kinematics(self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos))
            computed_target_pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.target_link_indices])

            # Position difference
            error = np.mean(np.linalg.norm(computed_target_pos - random_target_pos, axis=-1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization position error: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_vector_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.vector, hand_type)
        if config_path is None:
            return

        # Note: The parameters below are adjusted solely for this test
        # For retargeting from human to robot, their values should remain the default in the retargeting config
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["low_pass_alpha"] = 0
        override["scaling_factor"] = 1.0
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert retargeting.optimizer.retargeting_type == "VECTOR"

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = self.generate_vector_retargeting_data_gt(robot, optimizer)
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Optimized vector
            computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=fixed_qpos, last_qpos=init_qpos[:])

            # Check results
            robot.compute_forward_kinematics(self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos))
            computed_pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices])
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=-1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization vector error: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", DEXPILOT_ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType])
    def test_dexpilot_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, hand_type)
        if config_path is None:
            return

        # Note: The parameters below are adjusted solely for this test
        # For retargeting from human to robot, their values should remain the default in the retargeting config
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["low_pass_alpha"] = 0
        override["scaling_factor"] = 1.0
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()
        assert retargeting.optimizer.retargeting_type == "DEXPILOT"

        robot: RobotWrapper = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = self.generate_vector_retargeting_data_gt(robot, optimizer)
            fixed_qpos = random_qpos[optimizer.idx_pin2fixed]

            # Optimized vector
            computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=fixed_qpos, last_qpos=init_qpos[:])

            robot.compute_forward_kinematics(self.compute_pin_qpos(optimizer, computed_qpos, fixed_qpos))
            computed_pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices])
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=-1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization vector error for DexPilot retargeting: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2
