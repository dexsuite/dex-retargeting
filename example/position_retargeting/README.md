## Generate Robot Motion from DexYCB dataset

### Prepare DexYCB Dataset

This example shows that how to leverage the fantastic DexYCB dataset to generate robot motion trajectory.
[DexYCB](https://dex-ycb.github.io/) is a hand-object dataset made by NVIDIA. To run this demo, you need to download at
least one compressed files following the [original instruction](https://dex-ycb.github.io).

Now we use the subset `20200709-subject-01.tar.gz` from DexYCB as an example.

1. Download `20200709-subject-01.tar.gz` and place in it somewhere.
2. Download `models` and `calibration`, and place them next to the `20200709-subject-01.tar.gz`.

```Log
├── 20200709-subject-01
├── calibration
└── models
```

3. Check the download data using `dataset.py`. It will print the number of frames for each object.

```shell
export DEX_YCB_DIR=/path/to/dex-ycb # which contains the three calibration and models directory
python3 dataset.py
```

### Install manopth

Here we install the manopth in the same way as [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit).

1. Download manopth in this directory, the manopth should locate at `dex-robot-zoo/example/dex_ycb`

    ```shell
    git clone https://github.com/hassony2/manopth
    pip3 install chumpy opencv-python # install manopth dep
    ```

2. Download MANO models and local install manopth
   Download MANO models and code (`mano_v1_2.zip`) from the [MANO website](https://mano.is.tue.mpg.de) and place it
   inside `manopth`.

    ```shell
    cd manopth
    pip3 install -e .
    unzip mano_v1_2.zip
    cd mano
    ln -s ../mano_v1_2/models models
    ```

### Visualize interaction between human hand and object

Without retargeting, we can first visualize the original dataset in SAPIEN viewer. The hand mesh is computed via
manopth/

```shell
python3 visualize_hand_object.py --mode human
# Press q to quit viewer
```

### Visualize interaction between robot hand and object

```shell
python3 visualize_hand_object.py --mode adroit # for one robot
python3 visualize_hand_object.py --mode allegro # for multiple robot
# Press q to quit viewer
```






