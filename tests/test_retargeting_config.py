from pathlib import Path

import pytest

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
    def test_config_parsing(self, config_path):
        config_path = self.config_dir / config_path
        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()
        assert type(retargeting) == SeqRetargeting
