# EdgeSAM-UNet

Official PyTorch implementation of **EdgeSAM-UNet**, a lightweight semantic segmentation framework for road defect segmentation. The model builds a U-Net-style decoder on top of the EdgeSAM image encoder and introduces edge-aware and frequency-aware feature enhancement modules for accurate and efficient segmentation.

> This repository is intended for academic research and reproducible experiments on road defect segmentation or other small-object segmentation tasks.

---

## Highlights

- **Lightweight EdgeSAM encoder**: uses the image encoder from EdgeSAM as the backbone.
- **Parameter-efficient adaptation**: supports LoRA / MiLoRA-style 1×1 convolution adapters injected into the frozen backbone.
- **Edge-aware decoder**: introduces edge enhancement into convolutional decoding blocks.
- **Frequency-aware feature enhancement**: uses Fourier-domain feature refinement inside the EBFE module.
- **Deep supervision**: produces three segmentation outputs during training.
- **Deployment-oriented fusion**: provides RepVGG-style re-parameterization and Conv-BN fusion utilities for inference optimization.

---

## Project Structure

```text
EDGESAM-UNET/
├── dataset.py                         # Dataset loader and data augmentation
├── model/
│   └── edgesamABCMiloracpr.py          # Main EdgeSAM-UNet model, LoRA/MiLoRA, loss and fusion functions
├── tools/
│   ├── EBFE.py                         # Edge/Frequency-aware feature enhancement module
│   └── EdgeEnhancer.py                 # Edge enhancement module
├── train/
│   └── train.py                        # Training script
├── test/
│   ├── test.py                         # Evaluation, prediction saving, FPS/FLOPs calculation
│   └── compute_flops.py                # FLOPs/parameter calculation script
├── edge_sam/                           # EdgeSAM backbone implementation
├── sam2/                               # SAM2-related modules
├── edge_sam/weights/                   # EdgeSAM pretrained weights
└── sam2_hiera_tiny.pt                  # SAM2 checkpoint, if used
```

---

## Environment Requirements

The code is based on Python and PyTorch. A CUDA-enabled GPU is recommended for training.

Recommended environment:

```bash
conda create -n edgesam_unet python=3.10 -y
conda activate edgesam_unet
```

Install PyTorch according to your CUDA version, then install the remaining packages:

```bash
pip install torch torchvision
pip install numpy pillow matplotlib tqdm imageio medpy thop
```

Optional packages may be required depending on your local SAM/EdgeSAM configuration.

---

## Pretrained Weights

The model uses EdgeSAM pretrained weights as the encoder initialization. Place the checkpoint file under:

```text
edge_sam/weights/
```

For example:

```text
edge_sam/weights/edge_sam.pth
edge_sam/weights/edge_sam_3x.pth
```

During training, pass the checkpoint path through `--checkpoint_path`:

```bash
--checkpoint_path ./edge_sam/weights/edge_sam_3x.pth
```

> For public GitHub release, large checkpoint files are not recommended to be committed directly. Please provide an external download link or use Git LFS.

---

## Dataset Preparation

The dataset should be organized into separate image and mask folders. The current dataset loader matches images and masks by sorted file order.

A recommended structure is:

```text
data/
└── RoadDefectDataset/
    ├── train/
    │   ├── images/
    │   │   ├── 0001.jpg
    │   │   └── 0002.jpg
    │   └── labels/
    │       ├── 0001.png
    │       └── 0002.png
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Mask requirements:

- Mask files should be grayscale `.png` images.
- Pixel values should represent class indices, e.g. `0, 1, 2, 3` for a 4-class task.
- The number of classes should be consistent with `--num_classes`.
- Image and mask filenames should correspond after sorting.

---

## Training

Run training with:

```bash
python train/train.py \
  --save_path ./runs/edgesam_unet_exp1 \
  --model_type edge_sam \
  --checkpoint_path ./edge_sam/weights/edge_sam_3x.pth \
  --train_image_path ./data/RoadDefectDataset/train/images \
  --train_mask_path ./data/RoadDefectDataset/train/labels \
  --val_image_path ./data/RoadDefectDataset/val/images \
  --val_mask_path ./data/RoadDefectDataset/val/labels \
  --num_classes 4 \
  --ignore_class_index 255 \
  --epoch 200 \
  --lr 0.001 \
  --batch_size 16 \
  --weight_decay 0.0005
```

The training script saves:

```text
runs/edgesam_unet_exp1/
├── best_model.pth
├── log.txt
└── loss_curve.png
```

The model returns three outputs:

```python
pred0, pred1, pred2 = model(x)
```

The total training loss is computed as:

```python
loss = structure_loss(pred0, target) + structure_loss(pred1, target) + structure_loss(pred2, target)
```

`structure_loss` combines weighted cross-entropy and weighted IoU loss, which is suitable for road defect segmentation with small or irregular target regions.

---

## Evaluation

Run evaluation with:

```bash
python test/test.py \
  --checkpoint ./runs/edgesam_unet_exp1/best_model.pth \
  --save_path ./results/edgesam_unet_exp1 \
  --test_image_path ./data/RoadDefectDataset/test/images \
  --test_gt_path ./data/RoadDefectDataset/test/labels \
  --num_classes 4
```

The script reports:

- Dice Similarity Coefficient, DSC
- Intersection over Union, IoU
- Frames per second, FPS
- FLOPs
- Number of parameters

Prediction results are saved as:

```text
results/edgesam_unet_exp1/
├── xxx_gray.png     # grayscale predicted mask
└── xxx_color.png    # color visualization
```

---

## Model Overview

The main model is implemented in:

```text
model/edgesamABCMiloracpr.py
```

The architecture consists of:

1. **Frozen EdgeSAM image encoder**
   - The original prompt encoder and mask decoder are removed.
   - Multi-scale features are extracted from the EdgeSAM image encoder.

2. **LoRA / MiLoRA adapter injection**
   - 1×1 convolution adapters are injected into selected backbone stages.
   - The original backbone parameters are frozen.
   - Only lightweight adapter and decoder parameters are trained.

3. **EBFE module**
   - Enhances multi-scale features using spatial convolution, 3D convolution, 1D convolution and Fourier-domain refinement.

4. **Edge-enhanced U-Net decoder**
   - Uses skip connections from different encoder stages.
   - Edge enhancement is applied inside decoder convolution blocks.

5. **Deep supervision heads**
   - Intermediate outputs are generated to improve optimization stability.

---

## Inference Optimization

The project provides deployment-oriented fusion functions:

```python
from model.edgesamABCMiloracpr import fuse_repvgg_layers, fuse_model_conv_bn_full

fuse_repvgg_layers(model)
fuse_model_conv_bn_full(model)
```

These functions are used to fuse RepVGG-style blocks and Conv-BN structures before inference.

---

## Important Notes Before Public Release

Before releasing the repository on GitHub, it is recommended to clean the project:

```text
Remove:
- .idea/
- __pycache__/
- *.pyc
- local experiment logs
- local result folders
- private dataset paths
- large checkpoint files, unless using Git LFS
```

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.idea/
.vscode/
data/
datasets/
runs/
results/
checkpoints/
weights/
*.pth
*.pt
*.ckpt
*.onnx
*.log
```

Please also verify the command-line arguments before release. In particular, make sure local hard-coded paths in `train/train.py`, `test/test.py`, and `test/compute_flops.py` are replaced by command-line arguments or relative paths.

---



## Acknowledgement

This project builds upon the ideas and implementations of SAM, SAM2 and EdgeSAM. We sincerely thank the authors of the related open-source projects.

---


