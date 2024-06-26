
# 3D Shape Completion

See the [GENERATIVE MODELS FOR 3D SHAPE COMPLETION](https://drive.google.com/file/d/10-bWIvtXW_QMZn2NY1VrrYPnjzi7fgJG/view?usp=sharing) paper.

| || |
|---------|---------|---------|
| ![bed condition](./figs/32_64_bed_cond.png) | ![bed prediction](./figs/32_64_bed_pred.png) | ![bed ground truth](./figs/32_64_bed_gt.png) |

The repository contains code for training and sampling from a generative model for 3D shape completion. The model implemented in this repository is based on the diffusion model proposed in the paper [DiffComplete: Diffusion-based Generative 3D Shape Completion](https://arxiv.org/pdf/2306.16329.pdf).

**Credits**:

- base taken from [improved-diffusion](https://github.com/openai/improved-diffusion).

- evaluation part taken from [PatchComplete](https://github.com/yuchenrao/PatchComplete).

## Pretrained models

Pretrained models can be downloaded from [this link](https://drive.google.com/drive/folders/16Lqowbi5zq5eOGUsIo-sRDufkjS7pGL1?usp=sharing).

## Build environment

```bash
virtualenv -p python3.8 venv
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${pwd}"
pip install -r requirements.txt
```

**Note:** _CUDA_ is required to run the code (because of evaluation part).

## Data generation

To generate the dataset for shape completion script `dataset_hole.py` is used. To have same moddel used as in the paper, use `--filter_path` option to specify the path to the file with the list of models to be used for given dataset. The files are located in [./datasets/txt](./datasets/txt/) directory.

All avaiable arguments can be found by running `python ./dataset_hole.py --help`.

**Data sources:**

- [ShapeNet](https://github.com/yuchenrao/PatchComplete/tree/main?tab=readme-ov-file#download-processed-datasets)
- [ModelNet40](https://modelnet.cs.princeton.edu/)

### Shape completion dataset

To generate shape completion dataset run the following command:

```bash
cd dataset_processing
```

**Objaverse Furniture**

```bash
python ./dataset_hole.py --output  datasets/objaverse-furniture --tag_names chair lamp bathtub chandelier bench bed table sofa toilet
```

**Objaverse Vehicles**

```bash
python ./dataset_hole.py --output  datasets/objaverse-vehicles --category_names cars-vehicles --tag_names car truck bus airplane
```

**Objaverse Animals**

```bash
python ./dataset_hole.py --output  datasets/objaverse-animals --category_names animals-pets --tag_names cat dog
```

**ShapeNet**

```bash
python ./dataset_hole.py --dataset shapenet --source SHAPENET_DIR_PATH --output  datasets/shapenet
```

**ModelNet40**

```bash
python ./dataset_hole.py --dataset modelnet --source MODELNET40_DIR_PATH --output  datasets/modelnet40
```

### Super resolution dataset

Super resolution dataset used for training was created by running the shape completion model over traning and validation dataset to obtaion the predicted shapes, which where used as input for the super resolution model.

## Training

To train model the script `train.py` is used. All avaiable arguments can be found by running `python ./train.py --help`.

To train the **_BaseComplete_** model run the following command:

```bash
python ./scripts/train.py --batch_size 32 \
 --data_path "./datasets/objaverse-furniture/32/" \
 --train_file_path "../datasets/objaverse-furniture/train.txt" \
 --val_file_path "../datasets/objaverse-furniture/val.txt" \
 --dataset_name complete
```

to train with ROI mask add `--use_roi = True` option.

To train the **low res processing** model run the following command:

```bash
python ./scripts/train.py --batch_size 32 \
 ... # data options \
 --in_scale_factor 1 \
 --dataset_name complete_32_64
```

To train the **_superes_** model run the following command:

```bash
python ./scripts/train.py --batch_size 32 \
 --data_path "./datasets/objaverse-furniture-sr/" \
 --val_data_path "./datasets/objaverse-furniture-sr-val/" \
 --super_res True \
 --dataset_name sr
```

## Sampling

To sample one shape from the mesh model run the following command:

```bash
python ./scripts/sample.py --model_path MODEL_PATH \
 --sample_path SAMPLE_PATH \ # Condition
 --input_mesh True
 --condition_size 32 \ # Expected condition size
 --output_size 32 \ # Expected output size
```

or using .npy file as input:

```bash
python ./scripts/sample.py --model_path MODEL_PATH \
 --sample_path SAMPLE_PATH \ # Condition
 --input_mesh False
 --output_size 32 \ # Expected output size
```

## Evaluation

To evaluate on whole dataset run the following command:

```bash
python ./scripts/evaluate_dataset.py \
 --data_path "./datasets/objaverse-furniture/32"
 --file_path "./datasets/objaverse-furniture/test.txt"
 --model_path MODEL_PATH
```

## Results

Evaluation on _TEST_ dataset:
| Metric | ****BaseComplete**** | ****BaseComplete**** + **ROI mask** |
|---------|---------|---------|
| CD | 3.53 | 2.86 |
| IoU | 81.62 | 84.77 |
| L1 | 0.0264 | 0.0187 |

**Note:** CD and IoU are scaled by 100. Lower values are better for CD and L1, while higher values are better for IoU.

| Condition | Prediction | Ground Truth |
|---------|---------|---------|
| ![bathub condition](./figs/32_64_bathub_cond.png) | ![bathub predicted](./figs/32_64_bathub_pred.png) | ![bathub ground truth](./figs/32_64_bathub_gt.png) |
| ![couch condition](./figs/32_64_couch_cond.png) | ![couch prediction](./figs/32_64_couch_pred.png) | ![couch ground truth](./figs/32_64_couch_gt.png) |
| ![bed condition](./figs/32_64_bed_cond.png) | ![bed prediction](./figs/32_64_bed_pred.png) | ![bed ground truth](./figs/32_64_bed_gt.png) |
