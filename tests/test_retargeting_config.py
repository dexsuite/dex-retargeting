from pathlib import Path

import pytest
import yaml

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

VECTOR_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right.yml",
    "allegro_left": "teleop/allegro_hand_left.yml",
    "shadow_right": "teleop/shadow_hand_right.yml",
    "svh_right": "teleop/schunk_svh_hand_right.yml",
    "leap_right": "teleop/leap_hand_right.yml",
    "ability_right": "teleop/ability_hand_right.yml",
    "ability_left": "teleop/ability_hand_left.yml",
}
POSITION_CONFIG_DICT = {
    "allegro_right": "offline/allegro_hand_right.yml",
    "shadow_right": "offline/shadow_hand_right.yml",
    "svh_right": "offline/schunk_svh_hand_right.yml",
    "leap_right": "offline/leap_hand_right.yml",
    "ability_right": "offline/ability_hand_right.yml",
}
DEXPILOT_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right_dexpilot.yml",
    "allegro_left": "teleop/allegro_hand_left_dexpilot.yml",
    "shadow_right": "teleop/shadow_hand_right_dexpilot.yml",
    "svh_right": "teleop/schunk_svh_hand_right_dexpilot.yml",
    "leap_right": "teleop/leap_hand_right_dexpilot.yml",
}

ROBOT_NAMES = list(VECTOR_CONFIG_DICT.keys())


class TestRetargetingConfig:
    config_dir = Path(__file__).parent.parent / "dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir.absolute()))

    config_paths = (
        list(VECTOR_CONFIG_DICT.values()) + list(POSITION_CONFIG_DICT.values()) + list(DEXPILOT_CONFIG_DICT.values())
    )

    @pytest.mark.parametrize("config_path", config_paths)
    def test_path_config_parsing(self, config_path):
        config_path = self.config_dir / config_path
        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()
        assert isinstance(retargeting, SeqRetargeting)

    def test_dict_config_parsing(self):
        cfg_str = """
        type: position
        urdf_path: ability_hand/ability_hand_right.urdf
        wrist_link_name: "base_link"

        target_joint_names: ['index_q1', 'middle_q1', 'pinky_q1', 'ring_q1', 'thumb_q1', 'thumb_q2']
        target_link_names: [ "thumb_tip",  "index_tip", "middle_tip", "ring_tip", "pinky_tip" ]

        target_link_human_indices: [ 4, 8, 12, 16, 20 ]

        low_pass_alpha: 1
        """
        cfg_dict = yaml.safe_load(cfg_str)
        config = RetargetingConfig.from_dict(cfg_dict)
        retargeting = config.build()
        assert isinstance(retargeting, SeqRetargeting)

    def test_multi_dict_config_parsing(self):
        cfg_str = """
        - type: vector
          urdf_path: allegro_hand/allegro_hand_right.urdf
          wrist_link_name: "wrist"

          target_joint_names: null
          target_origin_link_names: [ "wrist", "wrist", "wrist", "wrist" ]
          target_task_link_names: [ "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ]
          scaling_factor: 1.6

          # The joint indices of human hand joint which corresponds to each link in the target_link_names
          target_link_human_indices: [ [ 0, 0, 0, 0 ], [ 4, 8, 12, 16 ] ]

          low_pass_alpha: 0.2

        - type: DexPilot
          urdf_path: leap_hand/leap_hand_right.urdf
          wrist_link_name: "base"

          target_joint_names: null
          finger_tip_link_names: [ "thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head" ]
          scaling_factor: 1.6

          low_pass_alpha: 0.2
        """
        cfg_dict_list = yaml.safe_load(cfg_str)
        retargetings = []
        for cfg_dict in cfg_dict_list:
            config = RetargetingConfig.from_dict(cfg_dict)
            retargeting = config.build()
            retargetings.append(retargeting)
            assert isinstance(retargeting, SeqRetargeting)

    @pytest.mark.parametrize("config_path", POSITION_CONFIG_DICT.values())
    def test_add_dummy_joint(self, config_path):
        config_path = self.config_dir / config_path
        override = {"add_dummy_free_joint": False}
        config = RetargetingConfig.load_from_file(config_path, override)
        retargeting = config.build()
        robot = retargeting.optimizer.robot
        original_robot_dof = robot.dof
        original_active_dof = len(retargeting.optimizer.target_joint_names)

        override = {"add_dummy_free_joint": True}
        config = RetargetingConfig.load_from_file(config_path, override)
        retargeting = config.build()
        robot = retargeting.optimizer.robot

        assert robot.dof == original_robot_dof + 6
        assert retargeting.joint_limits.shape == (original_active_dof + 6, 2)
        dummy_joint_names = robot.dof_joint_names[:6]
        for i in range(6):
            assert "dummy" in dummy_joint_names[i]
