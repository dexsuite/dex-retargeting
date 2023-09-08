## Retarget Robot Motion from Hand Object Pose Dataset

### Setting up DexYCB Dataset

This example illustrates how you can utilize the impressive DexYCB dataset to create a robot motion trajectory.
The [DexYCB](https://dex-ycb.github.io/) is a hand-object dataset developed by NVIDIA.
To execute this demonstration, you need to download at least one compressed file as per
the [official guidelines ↗](https://dex-ycb.github.io).

In this case, we will be using the `20200709-subject-01.tar.gz` subset from DexYCB.

1. Download `20200709-subject-01.tar.gz` and store it in a suitable location.
2. Download `models` and `calibration`, and keep them alongside the `20200709-subject-01.tar.gz`.

```Log
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

### Setting up manopth

Now, we will set up manopth similar to how it is done in [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit).

1. Download manopth in this directory, the manopth should be located
   at `dex_retargeting/example/position_retargeting/dex_ycb`

    ```shell
    git clone https://github.com/hassony2/manopth
    pip3 install chumpy opencv-python # install manopth dependencies
    ```

2. Download MANO models and locally install manopth
   Download MANO models and code (`mano_v1_2.zip`) from the [MANO website ↗](https://mano.is.tue.mpg.de) and place it
   inside `manopth`.

    ```shell
    cd manopth
    pip3 install -e .
    unzip mano_v1_2.zip
    cd mano
    ln -s ../mano_v1_2/models models
    ```

### Installing Additional Python Dependencies

```shell
pip install tyro pyyaml
```

### Visualizing Human Hand-Object Interaction

Before proceeding to retargeting, we can first visualize the original dataset in SAPIEN viewer. The hand mesh is
computed via manopth/

```shell
python3 visualize_hand_object.py --dexycb-dir=PATH_TO_YOUR_DEXYCB_DIR_ROOT
# Press q to exit viewer
```

### Visualizing Robot Hand-Object Interaction

Visualize the retargeting results for multiple robot hands along with the human hand.

```shell
python3 visualize_hand_object.py --dexycb-dir=PATH_TO_YOUR_DEXYCB_DIR_ROOT --robots allegro shadow svh
# Press q to exit viewer
```
