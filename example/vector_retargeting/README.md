## Retarget Robot Motion from Human Hand Video

### Generate the robot joint pose trajectory from our pre-recorded video

```shell
cd example/vector_retargeting
python3 detect_from_video.py \
  --robot-name allegro \
  --video-path data/human_hand_video.mp4 \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/allegro_joints.pkl 
```

This command will output the joint trajectory as a pickle file at the `output_path`.

The pickle file is a python dictionary with two keys: `meta_data` and `data`. `meta_data`, a dictionary, includes
details about the robot, while `data`, a list, contains the robotic joint positions for each frame. For additional
options, refer to the help information. Note that the time cost here includes both the hand pose detection from video,
and the hand pose retargeting in single process mode.

```shell
python3 detect_from_video.py --help
```

### Utilize the pickle file to produce a video of the robot

```shell
python3 render_robot_hand.py \
  --pickle-path data/allegro_joints.pkl \
  --output-video-path data/retargeted_allegro.mp4 \
  --headless
```

This command uses the data saved from the previous step to create a rendered video.

### Record a video of your own hand

```bash
python3 capture_webcam.py --video-path example/vector_retargeting/data/my_human_hand_video.mp4

```

This command will access your webcam (which should be connected to your computer) and record the video stream in mp4
format. To end video recording, press `q` on the keyboard.

### Retargeting from hand-object pose trajectory

Here we use the [DexYCB]() dataset to show that how can we retarget the human hand-object interaction trajectory to
robot hand-object interaction trajectory. After 
