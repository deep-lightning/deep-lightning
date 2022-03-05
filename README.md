# Deep Lightning [![Python 3.9.5](https://img.shields.io/badge/python-3.9.5-blue)](https://www.python.org/downloads/release/python-395/)

This project presents a neural network architecture based on cGANs (Conditional Generative Adversarial Networks) that seeks to approximate the indirect illumination of a scene.

This work is based on [Deep Illumination](https://github.com/CreativeCodingLab/DeepIllumination) and developed using [Pytorch Lightning](https://www.pytorchlightning.ai/).

## Installation

1. After installing [python](https://www.python.org/downloads/release/python-395/), create and activate a virtual environment
   ```
   > python -m venv <name_of_venv>
   > source <name_of_venv>/bin/activate
   ```
2. Inside the virtual environment install `pip-tools`
   ```
   > python -m pip install pip-tools
   ```
3. Compile production dependencies
   ```
   > pip-compile --allow-unsafe
   ```
4. (Optional) Compile dev dependencies
   ```
   > pip-compile --allow-unsafe dev-requirements.in
   ```
5. Install dependencies
   - Only production
     ```
     > pip-sync
     ```
   - Both production and dev
     ```
     > pip-sync requirements.txt dev-requirements.txt
     ```

## Dataset

In this project a dataset consists of a folder that contains as many subfolders as samples in it. Each subfolder should contain the G-buffers (depth map, diffuse map, normal map, local illumination) and the ground truths in case of training (global and indirect illumination).

An example of this structure would be the following

    <name_of_dataset>/
        <sample_1>/
            diffuse.hdr
            local.hdr
            normal.hdr
            z.hdr
            global.hdr (needed for training)
            indirect.hdr (needed for training)
        ...
        <sample_n>/
            diffuse.hdr
            local.hdr
            normal.hdr
            z.hdr
            global.hdr (needed for training)
            indirect.hdr (needed for training)

A dataset that follows the above structure can be created using the scripts we provide [here](https://github.com/deep-lightning/dataset-generator), which use [Radiance](https://www.radiance-online.org/) to render the different images.

### Available datasets

<!-- We also uploaded the datasets we created to -->

The datasets used in this project were also uploaded to [activeloop's hub](https://www.activeloop.ai/). They can be downloaded by running the following command

```
> python download.py <source> <target_folder>
```

where `<source>` is the name of the dataset to download and `<target_folder>` is where to download the dataset. If no target is specified, the same name as the source will be used.

> When using the datasets, in order to split them into train/val/test, an input argument named `data_regex` should be provided and chosen from a set of predefined regular expressions.

| Dataset name                                                             | Size | Description                                                                                                                          | Compatible data_regex   |
| ------------------------------------------------------------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| [vanilla](https:/app.activeloop.ai/deep-lightning/vanilla)               | 10k  | Cornell Box with an object (bunny, buddha, cube, dragon and sphere) placed in different positions                                    | vanilla, positions, all |
| [camera-variant](https:/app.activeloop.ai/deep-lightning/camera-variant) | 4k   | Cornell Box with both an object (bunny and buddha) and the camera placed in different positions                                      | cameras, all            |
| [light-variant](https:/app.activeloop.ai/deep-lightning/light-variant)   | 4k   | Cornell Box with both an object (bunny and buddha) and the light placed in different positions                                       | lights, all             |
| [wall-variant](https:/app.activeloop.ai/deep-lightning/wall-variant)     | 4k   | Cornell Box with two color combinations (red/greed and yellow/violet) and an object (bunny and buddha) placed in different positions | walls, all              |
| [object-variant](https:/app.activeloop.ai/deep-lightning/object-variant) | 4k   | Cornell Box with a cube and a sphere placed in different positions                                                                   | objects, all            |

## Usage

### Training a model

1. Activate virtual environment
   ```
   > source <name_of_venv>/bin/activate
   ```
2. Run `main.py` script in train mode, specifying the `dataset`, `data_regex` as well as any extra arguments
   ```
   > python main.py --train --dataset <path_to_folder> --data_regex <value>
   ```
   Extra arguments can be checked by doing
   ```
   > python main.py --help
   ```

The config used as well as metric logs can be then found under `lightning_logs/version_<x>/` folder. To view the logs with Tensorboard run

```
tensorboard --logdir lightning_logs
```

or

```
tensorboard --logdir lightning_logs/version_<x>/
```

### Testing a model

1. Activate virtual environment
   ```
   > source <name_of_venv>/bin/activate
   ```
2. Run `main.py` script in test mode, specifying the checkpoint to be used as well as any extra arguments

   ```
   > python main.py --test --ckpt <path_to_checkpoint>
   ```

   As before, extra arguments can be checked by doing

   ```
   > python main.py --help
   ```

### Inference with a model

A similar process to that of testing is followed

1. Activate virtual environment
   ```
   > source <name_of_venv>/bin/activate
   ```
2. Run `main.py` script in predict mode, specifying the checkpoint to be used and the input dataset as well as any extra arguments

   ```
   > python main.py --predict --ckpt <path_to_checkpoint> --dataset <path_to_folder>
   ```

   `<path_to_folder>` can be a dataset as specified in [Dataset](#dataset) or a single folder that contains the G-buffers.

   As before, extra arguments can be checked by doing

   ```
   > python main.py --help
   ```

## Authors

- Julio Morero
- Cecilia Gutiérrez
