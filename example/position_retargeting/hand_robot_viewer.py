from typing import Dict, List, Union

import numpy as np
import sapien.core as sapien
import transforms3d.quaternions

from dex_toolkit.retargeting.optimizer import PositionOptimizer
from dex_toolkit.retargeting.seq_retarget import SeqRetargeting
from dex_toolkit.sapien_utils.robot_object_utils import load_robot, LPFilter, modify_robot_visual, FREE_ROBOT_INFO_DICT, \
    STANDALONE_HAND_INTO_DICT
from hand_viewer import DatasetHandViewer

ROBOT2MANO = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])
ROBOT2MANO_POSE = sapien.Pose(q=transforms3d.quaternions.mat2quat(ROBOT2MANO))


def prepare_position_retargeting(joint_pos: np.array, link_hand_indices: np.ndarray):
    link_pos = joint_pos[link_hand_indices]
    return link_pos


def prepare_vector_retargeting(joint_pos: np.array, link_hand_indices_pairs: np.ndarray):
    joint_pos = joint_pos @ ROBOT2MANO
    origin_link_pos = joint_pos[link_hand_indices_pairs[0]]
    task_link_pos = joint_pos[link_hand_indices_pairs[1]]
    return task_link_pos - origin_link_pos


class DatasetHandRobotViewer(DatasetHandViewer):
    def __init__(self, gui, robot_name=""):
        super().__init__(gui)
        if robot_name:
            self.robots = [load_robot(self.scene, robot_name)]
            self.scene.step()
            self.filters = [LPFilter(10, 4)]
            self.robot_names = [robot_name]
            for robot in self.robots:
                modify_robot_visual(robot)
            self.robot_free = "free" in robot_name
            if self.robot_free:
                self.robot_info = FREE_ROBOT_INFO_DICT[robot_name]
            else:
                self.robot_info = STANDALONE_HAND_INTO_DICT[robot_name]

    def render_robot_data_frames(self, data: Dict, optimizer: SeqRetargeting,
                                 link_hand_indices: Union[List[int], List[List[int]]], fps=5):
        # Source DexYCB data
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        frame_num = hand_pose.shape[0]

        # Parse offset to render both the human and the robot hand
        y_offset = -1
        pose_offset = sapien.Pose(np.array([0, y_offset, 0]))
        print(f"Render data with {frame_num} frames")

        # Robot wrist offset
        wrist_name = self.robot_info.wrist_name
        wrist = [link for link in self.robots[0].get_links() if link.get_name() == wrist_name][0]
        relative_robot_wrist_pose = sapien.Pose(p=wrist.get_pose().p)

        # Position retargeting is only used as a globally, i.e., solve wrist pose and finger pose at the same time
        # Vector retargeting, including DexPilot retargeting is only used locally, i.e., solve finger pose only
        link_hand_indices = np.array(link_hand_indices)
        use_local_retargeting = not isinstance(optimizer.optimizer, PositionOptimizer)
        if use_local_retargeting:
            assert len(link_hand_indices.shape) == 2 and link_hand_indices.shape[0] == 2

        # Strip empty frame at the beginning
        start_frame = 0
        for i in range(0, frame_num):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self.compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        # Init object pose
        copy_num = len(self.robot_names) + 1
        actor_num = len(data["ycb_ids"])
        object_pose_frame = object_pose[start_frame]
        for k in range(actor_num):
            pos_quat = object_pose_frame[k]
            pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
            for copy_ind in range(copy_num):
                self.actors[k + copy_ind * actor_num].set_pose(pose)
                pose = pose_offset * pose

        # Loop simulation
        for i in range(start_frame, frame_num):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self.compute_hand_geometry(hand_pose_frame)
            # Visual hand and visual object
            for k in range(len(self.actors) // copy_num):
                pos_quat = object_pose_frame[k]
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.actors[k].set_pose(pose)
                for copy_ind in range(copy_num):
                    self.actors[k + copy_ind * actor_num].set_pose(pose)
                    pose = pose_offset * pose

            if vertex is not None:
                self.update_hand(vertex)

                if use_local_retargeting:
                    # Compute local MANO joint
                    local_hand_pose_frame = hand_pose_frame.copy()
                    local_hand_pose_frame[0, 0:3] = 0
                    local_hand_pose_frame[0, 48:51] = 0
                    _, local_joint = self.compute_hand_geometry(local_hand_pose_frame, use_camera_frame=True)

                    # Compute wrist global pose for MANO
                    axis_angle = hand_pose_frame[0, 0:3]
                    angle = np.linalg.norm(axis_angle)
                    axis = axis_angle / (angle + 1e-6)
                    quat = transforms3d.quaternions.axangle2quat(axis, angle)
                    global_wrist_quat = (self.camera_pose * sapien.Pose(q=quat) * ROBOT2MANO_POSE).q
                    global_wrist_pose = sapien.Pose(joint[0], q=global_wrist_quat)
                    root_pose = global_wrist_pose * relative_robot_wrist_pose.inv()
                    root_pose = pose_offset * root_pose
                    self.robots[0].set_pose(root_pose)

                    # Retargeting
                    local_joint[:, :] -= local_joint[0, :]
                    retargeting_input = prepare_vector_retargeting(local_joint, link_hand_indices)
                else:
                    retargeting_input = prepare_position_retargeting(joint, link_hand_indices)

                qpos = optimizer.retarget(retargeting_input, np.array([]))
                qpos = self.filters[0].next(qpos)
                qpos_viz = qpos.copy()
                if self.robot_free:
                    qpos_viz[1] += y_offset
                self.robots[0].set_qpos(qpos_viz)

            for k in range(fps):
                # Set viz robot
                self.scene.update_render()
                self.viewer.render()

        # self.viewer.toggle_pause(True)
        # self.viewer.render()

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]
        for _ in range(len(self.robot_names)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self.load_ycb_object(ycb_id, ycb_mesh_file)

    def compute_finger_tip_distance(self, human_joint):
        human_finger_pos_world = np.stack([human_joint[k] for k in [4, 8, 12, 16, 20]]).T  # (3, 5)
        human_finger_pos_homo = self.actors[0].get_pose().inv().to_transformation_matrix() @ np.concatenate(
            [human_finger_pos_world, np.ones([1, 5])])
        human_finger_pos = human_finger_pos_homo[:3, :].T  # (5,3)

        robot_link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        robot_link_pos = []
        for name in robot_link_names:
            link = [l for l in self.robots[0].get_links() if l.get_name() == name][0]
            robot_link_pos.append((self.actors[1].get_pose().inv() * link.get_pose()).p)

        mean_distance = np.linalg.norm(np.stack(robot_link_pos) - human_finger_pos, axis=1).mean()
        return mean_distance

#
# class DatasetHandMultiRobotViewer(DatasetHandRobotViewer):
#     def __init__(self, gui, robot_names: List[str]):
#         super().__init__(gui, robot_name="")
#         self.robots = [load_robot(self.scene, robot_name) for robot_name in robot_names]
#         for robot in self.robots:
#             modify_robot_visual(robot)
#         self.filters = [LPFilter(10, 4) for _ in robot_names]
#         self.robot_names = robot_names
#         self.num = 4
#
#     def render_robot_data_frames(self, data: Dict, optimizers: List[SeqRetargeting],
#                                  link_hands_indices: List[List[int]], fps=5):
#         link_hands_indices = [np.array(link_hand_indices) for link_hand_indices in link_hands_indices]
#         hand_pose = data["hand_pose"]
#         object_pose = data["object_pose"]
#         frame_num = hand_pose.shape[0]
#         y_offset = -0.3
#         pose_offset = sapien.Pose([0, y_offset, 0])
#
#         # Init hand pose
#         start_frame = 0
#         for i in range(0, frame_num):
#             init_hand_pose_frame = hand_pose[i]
#             vertex, joint = self.compute_hand_geometry(init_hand_pose_frame)
#             if vertex is not None:
#                 start_frame = i
#                 break
#         for robot_num in range(len(self.robot_names)):
#             qpos = optimizers[robot_num].retarget(joint[link_hands_indices[robot_num]], np.array([]))
#             qpos = self.filters[robot_num].next(qpos)
#             qpos[1] -= robot_num * y_offset
#             self.robots[robot_num].set_qpos(qpos)
#
#         # Init object pose
#         copy_num = len(self.robot_names) + 1
#         actor_num = len(data["ycb_ids"])
#         object_pose_frame = object_pose[start_frame]
#         for k in range(actor_num):
#             pos_quat = object_pose_frame[k]
#             pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
#             for copy_ind in range(copy_num):
#                 self.actors[k + copy_ind * actor_num].set_pose(pose)
#                 pose = pose_offset * pose
#
#         # Loop simulation
#         adroit_qpos = np.zeros(28)
#         for i in range(start_frame, frame_num):
#             object_pose_frame = object_pose[i]
#             hand_pose_frame = hand_pose[i]
#             vertex, joint = self.compute_hand_geometry(hand_pose_frame)
#             # Visual hand and visual object
#             for k in range(len(self.actors) // copy_num):
#                 pos_quat = object_pose_frame[k]
#                 pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
#                 self.actors[k].set_pose(pose)
#                 for copy_ind in range(copy_num):
#                     self.actors[k + copy_ind * actor_num].set_pose(pose)
#                     pose = pose_offset * pose
#
#             if vertex is not None:
#                 self.update_hand(vertex)
#                 for robot_num in range(len(self.robot_names)):
#                     qpos = optimizers[robot_num].retarget(joint[link_hands_indices[robot_num]], np.array([]))
#                     qpos = self.filters[robot_num].next(qpos).copy()
#                     qpos[1] += (robot_num + 1) * y_offset
#                     if robot_num == 0:
#                         self.robots[robot_num].set_qpos(qpos)
#                         adroit_qpos = qpos  # FF: 7,8,9, MF: 11, 12, 13, RF: 15,16,17, LF: 18, 20,21, 22 TH: 24, 25
#                     elif "allegro" in self.robot_names[robot_num]:
#                         qpos[np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])] = adroit_qpos[
#                             np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])]
#                         qpos[np.array([0, 2, 3, 4, 5])] = adroit_qpos[np.array([0, 2, 3, 4, 5])]
#                         qpos[1] = adroit_qpos[1] + y_offset * 2 - 0.08
#                         self.robots[robot_num].set_qpos(qpos)
#                     elif "dlr" in self.robot_names[robot_num]:
#                         index_dlr = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
#                         index_adroit = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22])
#                         qpos[index_dlr] = adroit_qpos[index_adroit] * 0.6
#                         qpos[np.array([0, 2, 3, 4, 5])] = adroit_qpos[np.array([0, 2, 3, 4, 5])]
#                         qpos[1] = adroit_qpos[1] + y_offset * 3 + 0.05
#                         qpos[0] += 0.03
#                         qpos[2] -= 0.01
#                         self.robots[robot_num].set_qpos(qpos)
#                     else:
#                         self.robots[robot_num].set_qpos(qpos)
#
#             for k in range(fps):
#                 # Set viz robot
#                 self.scene.update_render()
#                 self.viewer.render()
#
#         self.viewer.toggle_pause(True)
