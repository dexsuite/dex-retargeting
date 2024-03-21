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
}
POSITION_CONFIG_DICT = {
    "allegro_right": "offline/allegro_hand_right.yml",
    "shadow_right": "offline/shadow_hand_right.yml",
    "svh_right": "offline/schunk_svh_hand_right.yml",
    "leap_right": "offline/leap_hand_right.yml",
}
DEXPILOT_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right_dexpilot.yml",
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
        type: vector
        urdf_path: allegro_hand/allegro_hand_right.urdf
        wrist_link_name: "wrist"

        # Target refers to the retargeting target, which is the robot hand
        target_joint_names: null
        target_origin_link_names: [ "wrist", "wrist", "wrist", "wrist" ]
        target_task_link_names: [ "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ]
        scaling_factor: 1.6

        # Source refers to the retargeting input, which usually corresponds to the human hand
        # The joint indices of human hand joint which corresponds to each link in the target_link_names
        target_link_human_indices: [ [ 0, 0, 0, 0 ], [ 4, 8, 12, 16 ] ]

        # A smaller alpha means stronger filtering, i.e. more smooth but also larger latency
        low_pass_alpha: 0.2
        """
        cfg_dict = yaml.safe_load(cfg_str)
        config = RetargetingConfig.from_dict(cfg_dict)
        retargeting = config.build()
        assert type(retargeting) == SeqRetargeting

    def test_multi_dict_config_parsing(self):
        cfg_str = """
        - type: vector
          urdf_path: allegro_hand/allegro_hand_right.urdf
          wrist_link_name: "wrist"

          # Target refers to the retargeting target, which is the robot hand
          target_joint_names: null
          target_origin_link_names: [ "wrist", "wrist", "wrist", "wrist" ]
          target_task_link_names: [ "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ]
          scaling_factor: 1.6

          # Source refers to the retargeting input, which usually corresponds to the human hand
          # The joint indices of human hand joint which corresponds to each link in the target_link_names
          target_link_human_indices: [ [ 0, 0, 0, 0 ], [ 4, 8, 12, 16 ] ]

          # A smaller alpha means stronger filtering, i.e. more smooth but also larger latency
          low_pass_alpha: 0.2
          
        - type: DexPilot
          urdf_path: leap_hand/leap_hand_right.urdf
          wrist_link_name: "base"

          # Target refers to the retargeting target, which is the robot hand
          target_joint_names: null
          finger_tip_link_names: [ "thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head" ]
          scaling_factor: 1.6

          # A smaller alpha means stronger filtering, i.e. more smooth but also larger latency
          low_pass_alpha: 0.2
        """
        cfg_dict_list = yaml.safe_load(cfg_str)
        retargetings = []
        for cfg_dict in cfg_dict_list:
            config = RetargetingConfig.from_dict(cfg_dict)
            retargeting = config.build()
            retargetings.append(retargeting)
            assert isinstance(retargeting, SeqRetargeting)
