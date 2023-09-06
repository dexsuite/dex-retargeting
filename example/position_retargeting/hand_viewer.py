from typing import Dict, List, Optional

import numpy as np
import sapien.core as sapien
import torch
from sapien.core.pysapien import renderer as R
from sapien.utils import Viewer

from dataset import YCB_CLASSES
from dex_toolkit.sapien_utils.mesh_utils import compute_smooth_shading_normal_np
from mano_layer import MANOLayer


class DatasetHandViewer:
    def __init__(self, gui=True):
        # Setup
        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer(offscreen_only=not gui)
        sapien.VulkanRenderer.set_viewer_shader_dir("ibl")
        sapien.VulkanRenderer.set_camera_shader_dir("ibl")

        engine.set_renderer(renderer)
        config = sapien.SceneConfig()
        config.default_static_friction = 2.5
        config.default_dynamic_friction = 1.5
        config.default_restitution = 1e-1
        scene = engine.create_scene(config=config)
        scene.set_timestep(1 / 125)
        visual_material = renderer.create_material()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Lighting
        render_scene = scene.get_renderer_scene()
        scene.set_ambient_light(np.array([0.8, 0.8, 0.8]))
        scene.add_directional_light(np.array([1, -1, -1]), np.array([0.5, 0.5, 0.5]), shadow=True)
        scene.add_directional_light([0, 0, -1], [0.9, 0.8, 0.8], shadow=False)
        scene.add_spot_light(np.array([0, 0, 1.5]), direction=np.array([0, 0, -1]), inner_fov=0.3, outer_fov=1.0,
                             color=np.array([0.5, 0.5, 0.5]), shadow=False)

        # Viewer
        if gui:
            viewer = Viewer(renderer)
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1, 0, 1)
            viewer.set_camera_rpy(0, -0.6, 3.14)
            viewer.toggle_axes(0)
            self.viewer = viewer
        self.gui = gui

        # Table
        white_diffuse = renderer.create_material()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = scene.create_actor_builder()
        builder.add_box_collision(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.8, 2.0, 0.02]))
        builder.add_box_visual(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.8, 2.0, 0.02]), material=white_diffuse)
        builder.add_box_visual(sapien.Pose([0.7, 1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]),
                               material=white_diffuse)
        builder.add_box_visual(sapien.Pose([0.7, 1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]),
                               material=white_diffuse)
        builder.add_box_visual(sapien.Pose([0.7, -1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]),
                               material=white_diffuse)
        builder.add_box_visual(sapien.Pose([0.7, -1.9, -0.51]), half_size=np.array([0.015, 0.015, 0.49]),
                               material=white_diffuse)
        table = builder.build_static(name="table")

        # Caches
        self.engine = engine
        self.renderer = renderer
        self.scene = scene
        self.render_scene = render_scene
        self.internal_scene: R.Scene = scene.get_renderer_scene()._internal_scene
        self.context: R.Context = renderer._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        self.mano_layer: Optional[MANOLayer] = None
        self.mano_face: Optional[np.ndarray] = None
        self.camera_pose: Optional[sapien.Pose] = None
        self.actors: List[sapien.ActorBase] = []
        self.nodes: List[R.Node] = []

    def clear_all(self):
        for actor in self.actors:
            self.scene.remove_actor(actor)
        for _ in range(len(self.actors)):
            actor = self.actors.pop()
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
            self.load_ycb_object(ycb_id, ycb_mesh_file)

        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        self.camera_pose = sapien.Pose.from_transformation_matrix(extrinsic_mat).inv()

    def load_ycb_object(self, ycb_id, ycb_mesh_file):
        builder = self.scene.create_actor_builder()
        collision_file = ycb_mesh_file.replace("models", "convex_models").replace("textured_simple.obj",
                                                                                  "collision.obj")
        builder.add_multiple_collisions_from_file(collision_file)
        builder.add_visual_from_file(ycb_mesh_file)
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])
        self.actors.append(actor)

    def compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
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

    def update_hand(self, vertex):
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

    def render_data_frames(self, data: Dict, fps=10):
        if not self.gui:
            raise RuntimeError(f"Could not render data frames when the gui is disabled.")
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        frame_num = hand_pose.shape[0]

        step_per_frame = int(60 / fps)
        for i in range(frame_num):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, _ = self.compute_hand_geometry(hand_pose_frame)
            if vertex is not None:
                self.update_hand(vertex)
            for k in range(len(self.actors)):
                pos_quat = object_pose_frame[k]
                pose = self.camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
                self.actors[k].set_pose(pose)
            self.scene.update_render()
            for _ in range(step_per_frame):
                self.viewer.render()
        self.viewer.toggle_pause(True)
