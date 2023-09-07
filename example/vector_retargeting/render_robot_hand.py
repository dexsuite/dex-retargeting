from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
import tqdm
import tyro

from dex_retargeting.retargeting_config import RetargetingConfig


def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    import sapien.core as sapien
    from sapien.asset import create_dome_envmap
    from sapien.utils.viewer import Viewer

    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    engine = sapien.Engine()
    engine.set_log_level("warning")

    record_video = output_video_path is not None
    if not record_video:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 32
        sapien.render_config.rt_use_denoiser = True

    renderer = sapien.SapienRenderer(offscreen_only=headless)
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_directional_light([-1, 0.5, -1], color=[2.0, 2.0, 2.0], shadow=True, scale=2.0, shadow_map_size=4096)

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    robot = loader.load(config.urdf_path)
    robot_file_name = Path(config.urdf_path).stem
    if "shadow" in robot_file_name:
        robot.set_pose(sapien.Pose([0, 0, -0.3]))
    elif "svh" in robot_file_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "dlr" in robot_file_name:
        robot.set_pose(sapien.Pose([0, 0, -0.08]))
    else:
        robot.set_pose(sapien.Pose([0, 0, 0.0]))

    # Modify robot visual to make it pure white
    for actor in robot.get_links():
        for visual in actor.get_visual_bodies():
            for mesh in visual.get_render_shapes():
                mat = mesh.material
                mat.set_base_color(np.array([0.3, 0.3, 0.3, 1]))
                mat.set_specular(0.7)
                mat.set_metallic(0.1)

    cam = scene.add_camera(name="Cheese!", width=1280, height=720, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.313487, 0.0653831, -0.0111697], [0.088142, -0.0298786, -0.00264502, -0.995656]))

    # Setup onscreen viewer if not headless
    if not headless:
        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.focus_camera(cam)
        viewer.toggle_axes(False)
        viewer.toggle_camera_lines(False)
    else:
        viewer = None

    # Setup video recorder
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    for qpos in tqdm.tqdm(data):
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_texture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            # Use segmentation mask to paint background to white
            seg = cam.get_visual_actor_segmentation()[..., 0] < 1
            rgb[seg, :] = [255, 255, 255]
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    renderer_name: str = "sapien",
    output_video_path: Optional[str] = None,
    headless: bool = False,
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        renderer_name: Specifies the renderer to be used. It should be one from [sapien, sim_web_visualizer, trimesh].
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
    """
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    supported_renderer = ["sapien", "sim_web_visualizer", "trimesh"]

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    if renderer_name == "sapien":
        render_by_sapien(meta_data, data, output_video_path, headless)
    elif renderer_name == "sim_web_visualizer":
        raise NotImplementedError
    elif renderer_name == "trimesh":
        raise NotImplementedError
    else:
        raise ValueError(f"Renderer name should be one of the following: {supported_renderer}")


if __name__ == "__main__":
    tyro.cli(main)
