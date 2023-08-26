from pathlib import Path

import cv2
import numpy as np
import sapien.core as sapien
from sapien.asset import create_dome_envmap

from dex_retargeting.retargeting_config import get_retargeting_config, RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector

RECORD_VIDEO = False


def setup_sapien_viz_scene(urdf_path):
    from sapien.utils.viewer import Viewer
    import sapien.core as sapien

    engine = sapien.Engine()
    engine.set_log_level("warning")

    if not RECORD_VIDEO:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 32
        sapien.render_config.rt_use_denoiser = True

    renderer = sapien.SapienRenderer()

    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240)

    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_directional_light([-1, 0.5, -1], color=[2.0, 2.0, 2.0], shadow=True, scale=2.0, shadow_map_size=4096)

    loader = scene.create_urdf_loader()
    robot = loader.load(urdf_path)
    if robot_name == "shadow":
        robot.set_pose(sapien.Pose([0, 0, -0.3]))
    elif robot_name == "schunk":
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif robot_name == "dlr":
        robot.set_pose(sapien.Pose([0, 0, -0.08]))

    camera = scene.add_camera(name="photo", width=1280, height=720, fovy=1, near=0.1, far=10)
    camera.set_local_pose(
        sapien.Pose([0.313487, 0.0653831, -0.0111697], [0.088142, -0.0298786, -0.00264502, -0.995656])
    )

    if RECORD_VIDEO:
        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        # viewer.set_camera_pose(camera.pose)
    else:
        viewer = None

    for actor in robot.get_links():
        for visual in actor.get_visual_bodies():
            for mesh in visual.get_render_shapes():
                mat = mesh.material
                mat.set_base_color(np.array([0.3, 0.3, 0.3, 1]))
                mat.set_specular(0.7)
                mat.set_metallic(0.1)

    return scene, viewer


def build_retargeting(robot_name):
    if robot_name == "allegro":
        config_path = "teleop/allegro_hand_right.yml"
    elif robot_name == "shadow":
        config_path = "teleop/shadow_hand_right.yml"
    elif robot_name == "schunk":
        config_path = "teleop/schunk_svh_hand_right.yml"
    else:
        raise ValueError(f"Unrecognized robot_name: {robot_name}")

    test_config = get_retargeting_config(str(Path(__file__).parent.parent / f"dex_retargeting/configs/{config_path}"))
    seq_retargeting = test_config.build()
    return seq_retargeting, test_config


def retarget_video(seq_retargeting: SeqRetargeting, scene: sapien.Scene, viewer):
    video_path = Path(__file__).parent / "data/human_hand_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    robot = scene.get_all_articulations()[0]

    if not RECORD_VIDEO:
        writer = cv2.VideoWriter(f"data/output_{robot_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        detector = SingleHandDetector(hand_type="Right", selfie=False)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            rgb = frame[..., ::-1]
            num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)

            retargeting_type = seq_retargeting.optimizer.retargeting_type
            indices = seq_retargeting.optimizer.target_link_human_indices
            if retargeting_type == "VECTOR":
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            elif retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                raise ValueError(f"Unknown retargeting type: {retargeting_type}")
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos)
            if RECORD_VIDEO:
                for _ in range(3):
                    viewer.render()
            else:
                cam = scene.get_cameras()[0]
                scene.update_render()
                cam.take_picture()
                rgb = cam.get_texture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                seg = cam.get_visual_actor_segmentation()[..., 0] < 1
                rgb[seg, :] = [255, 255, 255]
                writer.write(rgb[..., ::-1])
                print(seq_retargeting.num_retargeting)

        cap.release()
        cv2.destroyAllWindows()
        if not RECORD_VIDEO:
            writer.release()


if __name__ == "__main__":
    robot_name = ["allegro", "shadow", "schunk"][1]
    robot_dir = Path(__file__).parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting, cfg = build_retargeting(robot_name)
    scene, viewer = setup_sapien_viz_scene(cfg.urdf_path)
    retarget_video(retargeting, scene, viewer)
