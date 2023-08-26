from pathlib import Path

import pytest

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from utils import VECTOR_CONFIG_DICT, POSITION_CONFIG_DICT, DEXPILOT_CONFIG_DICT


class TestRetargetingConfig:
    config_dir = Path(__file__).parent.parent / "dex_retargeting" / "configs"
    robot_dir = Path(__file__).parent.parent / "assets" / "robots"
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
