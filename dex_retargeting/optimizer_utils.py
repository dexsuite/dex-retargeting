from typing import Optional

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat


class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False


def add_dummy_free_joint(
    robot_builder: sapien.ArticulationBuilder,
    joint_indicator=(True, True, True, True, True, True),
    translation_range=(-1, 1),
    rotation_range=(-np.pi, np.pi),
):
    assert len(joint_indicator) == 6
    new_root = robot_builder.create_link_builder()
    parent = new_root

    # Prepare link and joint properties
    joint_types = ["prismatic"] * 3 + ["revolute"] * 3
    joint_limit = [translation_range] * 3 + [rotation_range] * 3
    joint_name = [f"dummy_{name}_translation_joint" for name in "xyz"] + [
        f"dummy_{name}_rotation_joint" for name in "xyz"
    ]
    link_name = [f"dummy_{name}_translation_link" for name in "xyz"] + [f"dummy_{name}_rotation_link" for name in "xyz"]

    # Find root link which has no parent
    root_link_builder = None
    for link_builder in robot_builder.get_link_builders():
        if link_builder.get_parent() == -1:
            root_link_builder = link_builder
            break
    assert root_link_builder is not None

    # Build free root
    valid_joint_num = 0
    for i in range(6):
        # Joint orders are x,y,z translation and then x,y,z rotation, 6 in total
        # If joint indicator for specific index is False, do not build this joint
        if not joint_indicator[i]:
            continue
        valid_joint_num += 1

        # Add small inertia property for more stable simulation
        parent.set_mass_and_inertia(1e-4, Pose(np.zeros(3)), np.ones(3) * 1e-6)
        parent.set_name(link_name[i])

        # The last joint will connect the last dummy link to the original root link of the robot
        if valid_joint_num < sum(joint_indicator):
            child = robot_builder.create_link_builder(parent)
        else:
            child = root_link_builder
            child.set_parent(parent.get_index())
        child.set_joint_name(joint_name[i])

        # Build joint
        if i == 3 or i == 0:
            child.set_joint_properties(joint_types[i], limits=np.array([joint_limit[i]]))
        elif i == 4 or i == 1:
            child.set_joint_properties(
                joint_types[i],
                limits=np.array([joint_limit[i]]),
                pose_in_child=Pose(q=euler2quat(0, 0, np.pi / 2)),
                pose_in_parent=Pose(q=euler2quat(0, 0, np.pi / 2)),
            )
        elif i == 2 or i == 5:
            child.set_joint_properties(
                joint_types[i],
                limits=np.array([joint_limit[i]]),
                pose_in_parent=Pose(q=euler2quat(0, -np.pi / 2, 0)),
                pose_in_child=Pose(q=euler2quat(0, -np.pi / 2, 0)),
            )
        parent = child

    return parent


class SAPIENKinematicsModelStandalone:
    def __init__(
        self, urdf_path, add_dummy_translation=False, add_dummy_rotation=False, scene: Optional[sapien.Scene] = None
    ):
        if scene is None:
            self.engine = sapien.Engine()
            self.scene = self.engine.create_scene()
        else:
            self.scene = scene
            self.engine = self.scene.engine
        loader = self.scene.create_urdf_loader()

        builder = loader.load_file_as_articulation_builder(urdf_path)
        if add_dummy_rotation or add_dummy_translation:
            dummy_joint_indicator = (add_dummy_translation,) * 3 + (add_dummy_rotation,) * 3
            add_dummy_free_joint(builder, dummy_joint_indicator)
        self.robot = builder.build(fix_root_link=True)
        self.robot.set_pose(sapien.Pose())
        self.robot.set_qpos(np.zeros(self.robot.dof))

    def release(self):
        self.scene = None
        self.engine = None
