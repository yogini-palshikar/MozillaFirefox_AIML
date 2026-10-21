# Waste Image Classification (TrashNet)

A computer-vision pipeline that classifies waste photos into six categories: **cardboard, glass, metal, paper, plastic, trash**. It compares a baseline CNN trained from scratch with a fine-tuned, ImageNet-pretrained ResNet18, and includes evaluation, error analysis, confidence-threshold tuning, latency measurement, and an inference script.

## Repository contents

| File | Purpose |
|---|---|
| `trashnet_waste_classification.ipynb` | Full pipeline: data prep, splits, training, evaluation, failure cases, threshold tuning, latency, demo |
| `predict.py` | Inference script: returns predicted class and confidence |
| `trashnet_resnet18.pt` | Trained ResNet18 weights (optional; created by the notebook) |
| `splits.json` | Saved train/val/test indices (seed 42) |
| `requirements.txt` | Python dependencies |
| `images/` | Plots used in this README (optional) |

## Dataset

- **Source:** TrashNet, Thung & Yang (Stanford). Original repo: https://github.com/garythung/trashnet
- **Mirror used:** https://huggingface.co/datasets/garythung/trashnet (MIT license)
- **Version:** Hugging Face commit `<PASTE COMMIT HASH PRINTED BY THE NOTEBOOK>`
- **Rows in the mirror:** `<N_RAW>` (about 2x the original 2,527 photos, because each photo appears to be stored in two resolutions)
- **Unique photos after de-duplication:** `<N_UNIQUE>`

### Images per class (after de-duplication)

| Class | Count | Share |
|---|---|---|
| cardboard | `<>` | `<>`% |
| glass | `<>` | `<>`% |
| metal | `<>` | `<>`% |
| paper | `<>` | `<>`% |
| plastic | `<>` | `<>`% |
| trash | `<>` | `<>`% |

### Class imbalance

`trash` is much smaller than the other classes (imbalance ratio largest/smallest = `<RATIO>`). Mitigations: stratified splits, class-weighted cross-entropy loss, and reporting per-class precision/recall/F1 and macro-F1 rather than accuracy alone.

### Duplicate handling

The mirror contains each photo more than once (original and resized copy). These are not byte-identical, so an exact hash is not enough. The notebook uses a perceptual hash (dHash, checked over 4 rotations, Hamming distance <= 6 of 64 bits) to keep one copy per photo **before** splitting. After splitting, it asserts that no near-duplicate appears across train, validation, or test.

## Splits

Stratified 70 / 15 / 15 with `random_state=42`. Indices are saved in `splits.json`.

| Split | Images |
|---|---|
| Train | `<>` |
| Validation | `<>` |
| Test | `<>` |

## Architecture

```
Image (any size)
   |  resize 224x224, normalise (ImageNet mean/std)
   v
+-----------------------------+      +--------------------------------------+
| Baseline CNN (from scratch) |      | ResNet18 (ImageNet-pretrained)       |
| 4 x [Conv3x3-BN-ReLU-Pool]  |      | conv1 -> layer1..layer4 -> avgpool   |
| 32-64-128-256 ch, GAP       |      | fc replaced: 512 -> 6                |
| Dropout 0.3 -> Linear(6)    |      | all layers fine-tuned                |
+--------------+--------------+      +------------------+-------------------+
               \_______________________  _______________/
                                       v
                          softmax -> class + confidence
                                       v
                  confidence >= threshold ? class : "uncertain"
```

## Training setup

- Optimiser: AdamW, cosine learning-rate schedule, weight decay 1e-4
- Loss: class-weighted cross-entropy (weights from the training split only)
- Baseline CNN: 20 epochs, lr 1e-3. ResNet18: 8 epochs, lr 3e-4
- Best epoch chosen by validation accuracy
- Augmentation: random resized crop (scale 0.7-1.0), horizontal flip, rotation (+/-20 degrees), colour jitter
- Seed: 42

## Results (test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline CNN + augmentation | `<>` | `<>` |
| ResNet18 + augmentation (final) | `<>` | `<>` |
| ResNet18, no augmentation | `<>` | `<>` |

**Transfer learning vs baseline:** `<one or two sentences describing the gap in your run>`

**Effect of augmentation:** `<describe the train-vs-val gap with and without augmentation, and the test difference>`

### Per-class metrics (final model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cardboard | `<>` | `<>` | `<>` | `<>` |
| glass | `<>` | `<>` | `<>` | `<>` |
| metal | `<>` | `<>` | `<>` | `<>` |
| paper | `<>` | `<>` | `<>` | `<>` |
| plastic | `<>` | `<>` | `<>` | `<>` |
| trash | `<>` | `<>` | `<>` | `<>` |

### Confusion matrix

![Confusion matrix](images/confusion_matrix.png)

## Failure cases and error analysis

![Failure cases](images/failure_cases.png)

Most common confusions in my run: `<e.g. glass -> plastic: N, paper -> cardboard: N>`

Likely causes:

- **Visually similar materials:** transparent, shiny glass and plastic; paper and cardboard share fibre texture.
- **Reflective metal:** cans mirror their surroundings and can resemble glass or plastic.
- **`trash` is a catch-all class** with few examples and no consistent appearance.
- **Dataset bias:** single objects on a plain background under similar lighting; crumpled, dirty, or partly hidden items fall outside this distribution.
- **Resizing to 224x224** distorts the original 4:3 aspect ratio slightly.

Per-class error analysis: `<which classes have the lowest recall and why>`

## Confidence-threshold tuning

Predictions below a confidence threshold are returned as `"uncertain"`. The threshold was chosen on the validation set (smallest threshold reaching `<TARGET>`% accuracy on accepted images), then evaluated on the test set.

| Threshold | Test coverage | Test accuracy on accepted |
|---|---|---|
| `<chosen>` | `<>` | `<>` |

## Inference latency (batch size 1, model only)

| Model | CPU mean / p95 (ms) | GPU mean / p95 (ms) |
|---|---|---|
| Baseline CNN | `<>` | `<>` |
| ResNet18 | `<>` | `<>` |

Hardware: `<CPU / GPU used>`. Latency excludes image loading and preprocessing.

## How to run

### Train and evaluate

1. Open `trashnet_waste_classification.ipynb` in Google Colab (Runtime -> GPU recommended) or locally with Jupyter.
2. Run all cells top to bottom (about 10-15 minutes on a T4, plus several minutes for dataset download and de-duplication).
3. The notebook writes `trashnet_resnet18.pt`, `splits.json`, and `predict.py`.

### Inference

```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg            # uses threshold saved with the model
python predict.py path/to/image.jpg 0.7        # custom threshold
```

Example output:

```
{'class': 'plastic', 'top_class': 'plastic', 'confidence': 0.9312, 'probabilities': {...}}
```

From Python:

```python
from predict import load_model, predict

model, classes, thr, size = load_model("trashnet_resnet18.pt")
print(predict("photo.jpg", model, classes, size, thr))
```

## Limitations for real campus waste

- **Domain gap:** TrashNet photos show one clean object on a white background. Real bins have clutter, multiple items, food residue, bags, poor lighting, and occlusion, so accuracy will be lower than the test numbers above.
- **Small, imbalanced data:** roughly 2.5k unique photos; `trash` is the smallest class. With a test set of a few hundred images, one image moves accuracy by about 0.3 percentage points, so differences of a point or two between models are not reliable.
- **Missing categories:** no e-waste, organic/food waste, Tetra Pak, batteries, or composite items such as plastic-lined coffee cups.
- **Single label per image:** no object detection, so photos with several items cannot be handled.
- **Calibration:** softmax confidence from fine-tuned networks tends to be over-confident. Use the "uncertain" route to a human, and ideally collect campus photos to fine-tune and recalibrate.

## Possible improvements

- Collect and label campus-specific photos; fine-tune on them.
- Add an object detector for multi-item scenes.
- Try larger backbones (ResNet50, EfficientNet) and temperature scaling for calibration.

## Reproducibility

Seed 42 is used for splits, weight initialisation, and data loading. GPU nondeterminism may cause small run-to-run differences in metrics.

## Credits and license

- Dataset: Gary Thung and Mindy Yang, TrashNet (https://github.com/garythung/trashnet), MIT license.
- Pretrained weights: torchvision ResNet18 (ImageNet).
