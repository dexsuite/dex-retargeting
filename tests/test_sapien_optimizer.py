from pathlib import Path
from time import time

import numpy as np
import pytest
import sapien.core as sapien

from dex_retargeting.constants import ROBOT_NAMES, get_default_config_path, RetargetingType, HandType
from dex_retargeting.optimizer import VectorOptimizer, PositionOptimizer
from dex_retargeting.retargeting_config import RetargetingConfig


class TestSapienOptimizer:
    np.set_printoptions(precision=4)
    config_dir = Path(__file__).parent.parent / "dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))

    @staticmethod
    def generate_vector_retargeting_data_gt(robot: sapien.Articulation, optimizer: VectorOptimizer):
        joint_limit = robot.get_qlimits()
        random_qpos = np.random.uniform(joint_limit[:, 0], joint_limit[:, 1])
        robot.set_qpos(random_qpos)

        random_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
        origin_pos = random_pos[optimizer.origin_link_indices]
        task_pos = random_pos[optimizer.task_link_indices]
        random_target_vector = task_pos - origin_pos
        init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.5, joint_limit[:, 0], joint_limit[:, 1])

        return random_qpos, init_qpos, random_target_vector

    @staticmethod
    def generate_position_retargeting_data_gt(robot: sapien.Articulation, optimizer: PositionOptimizer):
        joint_limit = robot.get_qlimits()
        random_qpos = np.random.uniform(joint_limit[:, 0], joint_limit[:, 1])
        robot.set_qpos(random_qpos)

        random_target_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
        init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.5, joint_limit[:, 0], joint_limit[:, 1])

        return random_qpos, init_qpos, random_target_pos

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType][:1])
    def test_position_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)

        # Note: The parameters below are adjusted solely for this test
        # The smoothness penalty is deactivated here, meaning no low pass filter and no continuous joint value
        # This is because the test is focused solely on the efficiency of single step optimization
        override = dict()
        override["normal_delta"] = 0
        config = RetargetingConfig.load_from_file(config_path, override)

        retargeting = config.build()

        robot = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random position
            random_qpos, init_qpos, random_target_pos = self.generate_position_retargeting_data_gt(robot, optimizer)

            # Optimized position
            computed_qpos = optimizer.retarget(random_target_pos, fixed_qpos=[], last_qpos=init_qpos[:])
            robot.set_qpos(np.array(computed_qpos))
            computed_target_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])

            # Position difference
            error = np.mean(np.linalg.norm(computed_target_pos - random_target_pos, axis=1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization position error: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType][:1])
    def test_vector_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.vector, hand_type)

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

        robot = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = self.generate_vector_retargeting_data_gt(robot, optimizer)

            # Optimized vector
            computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=[], last_qpos=init_qpos[:])
            robot.set_qpos(np.array(computed_qpos))
            computed_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization vector error: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2

    @pytest.mark.parametrize("robot_name", ROBOT_NAMES)
    @pytest.mark.parametrize("hand_type", [name for name in HandType][:1])
    def test_dexpilot_optimizer(self, robot_name, hand_type):
        config_path = get_default_config_path(robot_name, RetargetingType.dexpilot, hand_type)

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

        robot = retargeting.optimizer.robot
        optimizer = retargeting.optimizer

        num_optimization = 100
        tic = time()
        errors = dict(pos=[], joint=[])
        np.random.seed(1)
        for i in range(num_optimization):
            # Sampled random vector
            random_qpos, init_qpos, random_target_vector = self.generate_vector_retargeting_data_gt(robot, optimizer)

            # Optimized vector
            computed_qpos = optimizer.retarget(random_target_vector, fixed_qpos=[], last_qpos=init_qpos[:])
            robot.set_qpos(np.array(computed_qpos))
            computed_pos = np.array([robot.get_links()[i].get_pose().p for i in optimizer.robot_link_indices])
            computed_origin_pos = computed_pos[optimizer.origin_link_indices]
            computed_task_pos = computed_pos[optimizer.task_link_indices]
            computed_target_vector = computed_task_pos - computed_origin_pos

            # Vector difference
            error = np.mean(np.linalg.norm(computed_target_vector - random_target_vector, axis=1))
            errors["pos"].append(error)

        tac = time()
        print(f"Mean optimization vector error for DexPilot retargeting: {np.mean(errors['pos'])}")
        print(f"Retargeting computation for {robot_name.name} takes {tac - tic}s for {num_optimization} times")
        assert np.mean(errors["pos"]) < 1e-2
