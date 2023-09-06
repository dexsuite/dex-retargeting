Dex Retargeting
---
<p align="center">
    <!-- code check badges -->
    <a href='https://github.com/dexsuite/dex-retargeting/blob/main/.github/workflows/test.yml'>
        <img src='https://github.com/dexsuite/dex-retargeting/actions/workflows/test.yml/badge.svg' alt='Test Status' />
    </a>
    <!-- license badge -->
    <a href="https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
    </a>
</p>

## Installation

```shell
pip3 install -e ".[example]"
# If you do not need to run the examples:
# pip install -e .

```

## Examples

### Retargeting from human hand video

1. **Generate the robot joint pose trajectory from our pre-recorded video.**

```shell
export PYTHONPATH=$PYTHONPATH:`pwd`
python3 example/vector_retargeting/detect_from_video.py \
  --robot-name allegro \
  --video-path example/vector_retargeting/data/human_hand_video.mp4 \
  --retargeting-type vector \
  --hand-type right \
  --output-path example/vector_retargeting/data/allegro_joints.pkl 
```

This command will output the joint trajectory as a pickle file at the `output_path`.

The pickle file is a python dictionary with two keys: `meta_data` and `data`. `meta_data`, a dictionary, includes
details about the robot, while `data`, a list, contains the robotic joint positions for each frame. For additional
options, refer to the help information. Note that the time cost here includes both the hand pose detection from video,
and the hand pose retargeting in single process mode.

```shell
python3 example/vector_retargeting/detect_from_video.py --help
```

2. **Utilize the pickle file to produce a video of the robot**

```shell
export PYTHONPATH=$PYTHONPATH:`pwd`
python3 example/vector_retargeting/render_robot_hand.py \
  --pickle-path example/vector_retargeting/data/allegro_joints.pkl \
  --output-video-path example/vector_retargeting/data/retargeted_allegro.mp4 \
  --headless
```

This command uses the data saved from the previous step to create a rendered video.

3. **Record a video of your own hand**

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 example/vector_retargeting/capture_webcam.py --video-path example/vector_retargeting/data/my_human_hand_video.mp4

```

This command will access your webcam (which should be connected to your computer) and record the video stream in mp4
format. To end video recording, press `q` on the keyboard.

### Retargeting from hand-object pose trajectory

Here we use the [DexYCB]() dataset to show that how can we retarget the human hand-object interaction trajectory to
robot hand-object interaction trajectory. After 
