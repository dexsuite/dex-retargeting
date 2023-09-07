import sapien.core as sapien
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from typing import Union

import numpy as np
import yaml

from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.seq_retarget import SeqRetargeting


@dataclass
class RetargetingConfig:
    type: str
    urdf_path: str
    add_dummy_free_joint: bool = False

    # Source refers to the retargeting input, which usually corresponds to the human hand
    # The joint indices of human hand joint which corresponds to each link in the target_link_names

    target_link_human_indices: Optional[np.ndarray] = None

    # Position retargeting link names
    target_link_names: Optional[List[str]] = None

    # Vector retargeting link names
    target_joint_names: Optional[List[str]] = None
    target_origin_link_names: Optional[List[str]] = None
    target_task_link_names: Optional[List[str]] = None

    # DexPilot retargeting link names
    finger_tip_link_names: Optional[List[str]] = None
    wrist_link_name: Optional[str] = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: float = 1.0

    # Optimization hyperparameter
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # Joint limit tag
    has_joint_limits: bool = True

    # Low pass filter
    low_pass_alpha: float = 0.1

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"

    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        # Vector retargeting requires: target_origin_link_names + target_task_link_names
        # Position retargeting requires: target_link_names
        if self.type == "vector":
            if self.target_origin_link_names is None or self.target_task_link_names is None:
                raise ValueError(f"Vector retargeting requires: target_origin_link_names + target_task_link_names")
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError(f"Vector retargeting origin and task links dim mismatch")
            if self.target_link_human_indices.shape != (2, len(self.target_origin_link_names)):
                raise ValueError(f"Vector retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices is None:
                raise ValueError(f"Vector retargeting requires: target_link_human_indices")

        elif self.type == "position":
            if self.target_link_names is None:
                raise ValueError(f"Position retargeting requires: target_link_names")
            self.target_link_human_indices = self.target_link_human_indices.squeeze()
            if self.target_link_human_indices.shape != (len(self.target_link_names),):
                raise ValueError(f"Position retargeting link names and link indices dim mismatch")
            if self.target_link_human_indices is None:
                raise ValueError(f"Position retargeting requires: target_link_human_indices")

        elif self.type == "dexpilot":
            if self.finger_tip_link_names is None or self.wrist_link_name is None:
                raise ValueError(f"Position retargeting requires: finger_tip_link_names + wrist_link_name")

        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path], override: Optional[Dict] = None):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            if "target_link_human_indices" in cfg:
                cfg["target_link_human_indices"] = np.array(cfg["target_link_human_indices"])
            if override is not None:
                for key, value in override.items():
                    cfg[key] = value
            config = RetargetingConfig(**cfg)
        return config

    def build(self, scene: Optional[sapien.Scene] = None) -> SeqRetargeting:
        from dex_retargeting.optimizer import (
            VectorOptimizer,
            PositionOptimizer,
            DexPilotAllegroOptimizer,
        )
        from dex_retargeting.optimizer_utils import SAPIENKinematicsModelStandalone
        from dex_retargeting import yourdfpy as urdf
        import tempfile

        # Process the URDF with yourdfpy to better find file path
        robot_urdf = urdf.URDF.load(self.urdf_path)
        urdf_name = self.urdf_path.split("/")[-1]
        temp_dir = tempfile.mkdtemp(prefix="teleop-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)
        sapien_model = SAPIENKinematicsModelStandalone(
            temp_path,
            add_dummy_translation=self.add_dummy_free_joint,
            add_dummy_rotation=self.add_dummy_free_joint,
            scene=scene,
        )
        robot = sapien_model.robot
        joint_names = (
            self.target_joint_names
            if self.target_joint_names is not None
            else [joint.get_name() for joint in robot.get_active_joints()]
        )
        if self.type == "position":
            optimizer = PositionOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            optimizer = DexPilotAllegroOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                scaling=self.scaling_factor,
            )
        else:
            raise RuntimeError()

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
        )
        # TODO: hack here for SAPIEN
        retargeting.scene = sapien_model.scene
        return retargeting


def get_retargeting_config(config_path) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config


if __name__ == "__main__":
    # Path below is relative to this file

    test_config = get_retargeting_config(str(Path(__file__).parent / "configs/allegro_hand.yml"))
    print(test_config)
    opt = test_config.build()
    print(opt.optimizer.target_link_human_indices)
