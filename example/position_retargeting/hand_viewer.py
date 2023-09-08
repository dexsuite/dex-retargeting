from typing import Dict, List, Optional

import numpy as np
import sapien.core as sapien
import torch
from sapien.asset import create_dome_envmap
from sapien.core.pysapien import renderer as R
from sapien.utils import Viewer

from dataset import YCB_CLASSES
from mano_layer import MANOLayer


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class HandDatasetSAPIENViewer:
    def __init__(self, headless=False, use_ray_tracing=False):
        # Setup
        engine = sapien.Engine()
        engine.set_log_level("warning")

        if use_ray_tracing:
            sapien.render_config.camera_shader_dir = "rt"
            sapien.render_config.viewer_shader_dir = "rt"
            sapien.render_config.rt_samples_per_pixel = 32
            sapien.render_config.rt_use_denoiser = True
        else:
            sapien.render_config.camera_shader_dir = "ibl"
            sapien.render_config.viewer_shader_dir = "ibl"

        renderer = sapien.SapienRenderer(offscreen_only=headless)
        engine.set_renderer(renderer)
        renderer.set_log_level("error")

        # Scene
        scene_config = sapien.SceneConfig()
        scene = engine.create_scene(scene_config)
        scene.set_timestep(1 / 240)

        # Lighting
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_directional_light(np.array([1, -1, -1]), np.array([1, 1, 1]), shadow=True)
        scene.add_directional_light([0, 0, -1], [0.9, 0.8, 0.8], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))
        scene.add_spot_light(
            np.array([0, 0, 1.5]),
            direction=np.array([0, 0, -1]),
            inner_fov=0.3,
            outer_fov=1.0,
            color=np.array([0.5, 0.5, 0.5]),
            shadow=False,
        )

        # Add ground
        visual_material = renderer.create_material()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Viewer
        if not headless:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.6, 3.14)
            viewer.toggle_axes(0)
            self.viewer = viewer
        self.gui = not headless

        # Table
        white_diffuse = renderer.create_material()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = scene.create_actor_builder()
        builder.add_box_collision(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]))
        builder.add_box_visual(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]), material=white_diffuse)
        builder.add_box_visual(
            sapien.Pose([0.4, 1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]), material=white_diffuse
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, 1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]), material=white_diffuse
        )
        builder.add_box_visual(
            sapien.Pose([0.4, -1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]), material=white_diffuse
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, -1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]), material=white_diffuse
        )
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

        # Caches
        self.engine = engine
        self.renderer = renderer
        self.scene = scene
        self.internal_scene: R.Scene = scene.get_renderer_scene()._internal_scene
        self.context: R.Context = renderer._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        self.mano_layer: Optional[MANOLayer] = None
        self.mano_face: Optional[np.ndarray] = None
        self.camera_pose: Optional[sapien.Pose] = None
        self.objects: List[sapien.ActorStatic] = []
        self.nodes: List[R.Node] = []

    def clear_all(self):
        for actor in self.objects:
            self.scene.remove_actor(actor)
        for _ in range(len(self.objects)):
            actor = self.objects.pop()
            self.scene.remove_actor(actor)
        self.clear_node()
        self.mano_layer = None

    def clear_node(self):
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)

    def load_object_hand(self, data: Dict):
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            self._load_ycb_object(ycb_id, ycb_mesh_file)

        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        self.camera_pose = sapien.Pose.from_transformation_matrix(extrinsic_mat).inv()

    def _load_ycb_object(self, ycb_id, ycb_mesh_file):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(ycb_mesh_file)
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])
        self.objects.append(actor)

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        # pose parameters all zero, no hand is detected
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]
        joint = joint.cpu().numpy()[0]
        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

    def _update_hand(self, vertex):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def render_dexycb_data(self, data: Dict, fps=10):
        if not self.gui:
            raise RuntimeError(f"Could not render data frames when the gui is disabled.")
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        frame_num = hand_pose.shape[0]

        step_per_frame = int(60 / fps)
        for i in range(frame_num):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, _ = self._compute_hand_geometry(hand_pose_frame)
            if vertex is not None:
                self._update_hand(vertex)
            for k in range(len(self.objects)):
                pos_quat = object_pose_frame[k]
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.objects[k].set_pose(pose)
            self.scene.update_render()
            for _ in range(step_per_frame):
                self.viewer.render()

        self.viewer.render()
        self.viewer.toggle_pause(True)
