## Retarget Robot Motion from Hand Object Pose Dataset

![teaser](hand_object.webp)

### Setting up DexYCB Dataset

This example illustrates how you can utilize the impressive DexYCB dataset to create a robot motion trajectory.
The [DexYCB](https://dex-ycb.github.io/) is a hand-object dataset developed by NVIDIA.
To execute this demonstration, you need to download at least one compressed file as per
the [official guidelines ↗](https://dex-ycb.github.io).

In this case, we will be using the `20200709-subject-01.tar.gz` subset from DexYCB.

1. Download `20200709-subject-01.tar.gz` and store it in a suitable location.
2. Download `models` and `calibration`, and keep them alongside the `20200709-subject-01.tar.gz`.

```Log
.
├── 20200709-subject-01
├── calibration
└── models
```

3. Verify the downloaded data using `dataset.py`. It will display the trajectory count for each object.
   The `PATH_TO_YOUR_DEXYCB_DIR_ROOT` should be the directory containing the three subfolders from the previous step

```shell
cd example/position_retargeting
python dataset.py --dexycb-dir=PATH_TO_YOUR_DEXYCB_DIR_ROOT
```

You will get something similar like this:

```shell
50
Counter({'002_master_chef_can': 12, '005_tomato_soup_can': 9, '004_sugar_box': 6, '003_cracker_box': 6, '008_pudding_box': 4, '006_mustard_bottle': 4, '009_gelatin_box': 3, '007_tuna_fish_can': 2, '019_pitcher_base': 1, '024_bowl': 1, '021_bleach_cleanser': 1, '010_potted_meat_can': 1})
dict_keys(['hand_pose', 'object_pose', 'extrinsics', 'ycb_ids', 'hand_shape', 'object_mesh_file', 'capture_name'])
```

### Setting up manopth

Now, we will set up manopth similar to how it is done in [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit).

1. Download manopth in this directory, the manopth should be located
   at `dex_retargeting/example/position_retargeting`

    ```shell
    git clone https://github.com/hassony2/manopth
    pip install chumpy opencv-python # install manopth dependencies
    ```

2. Download MANO models and locally install manopth
   Download MANO models and code (`mano_v1_2.zip`) from the [MANO website ↗](https://mano.is.tue.mpg.de) and place it
   inside `manopth`.

    ```shell
    cd manopth
    pip install -e .
    unzip mano_v1_2.zip
    cd mano
    ln -s ../mano_v1_2/models models
    ```

### Installing Additional Python Dependencies

```shell
pip install tyro pyyaml sapien==3.0.0b0
```

### Visualizing Human Hand-Object Interaction

Before proceeding to retargeting, we can first visualize the original dataset in SAPIEN renderer. The hand mesh is
computed via manopth.

```shell
python visualize_hand_object.py --dexycb-dir=PATH_TO_YOUR_DEXYCB_DIR_ROOT
# Close the viewer window to quit
```

### Visualizing Robot Hand-Object Interaction

Visualize the retargeting results for multiple robot hands along with the human hand.

```shell
python visualize_hand_object.py --dexycb-dir=PATH_TO_YOUR_DEXYCB_DIR_ROOT --robots allegro shadow svh
# Close the viewer window to quit
```
