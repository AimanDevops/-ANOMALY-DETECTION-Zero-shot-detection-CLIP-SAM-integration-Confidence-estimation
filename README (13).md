# 🫁 COVID-19 Anomaly Detection — Zero-Shot CLIP + SAM

A zero-shot medical anomaly detection system for COVID-19 chest X-rays using **BiomedCLIP** and **Segment Anything Model (SAM)**. No task-specific training required — the pipeline uses natural language prompts to localize and classify anomalous lung regions directly from images.

---

## 📌 Overview

This project combines two powerful vision-language and segmentation models to detect COVID-19 patterns in chest X-rays without any labeled training data:

- **BiomedCLIP** (Microsoft) — computes gradient-based attention maps from text prompts describing COVID-19 pathology
- **SAM ViT-H** (Meta) — refines detected regions into precise segmentation masks
- **Confidence Estimation** — blends masked-region CLIP similarity with global image similarity into a final anomaly score

The pipeline classifies images as **Normal**, **Mild**, **Moderate**, or **Severe** based on a tunable confidence threshold.

---

## 🏗️ Pipeline Architecture

```
Chest X-Ray Input
       │
       ▼
BiomedCLIP Attention Maps
   ├── COVID prompts  ──► anomaly attention map
   └── Normal prompt  ──► normal attention map
       │
       ▼
Differential Attention  (anomaly − normal)
       │
       ▼
Connected Component Detection  ──► candidate anomaly points
       │
       ▼
SAM Segmentation  ──► refined binary mask
       │
       ▼
Masked-Region CLIP Re-scoring
       │
       ▼
Final Confidence Score  (0.85 × refined + 0.15 × global)
       │
       ▼
Severity Classification: Normal / Mild / Moderate / Severe
```

---

## ✨ Features

- **Zero-shot detection** — no fine-tuning or labeled data needed
- **CLIP → SAM integration** — attention-guided point prompts feed SAM for precise masks
- **Differential attention** — subtracts normal lung attention to isolate pathological regions
- **Severity grading** — four-level severity output from a single confidence score
- **Balanced evaluation** — automatic class balancing to prevent majority-class bias
- **Standard metrics** — Accuracy, Recall, F1-Score, and ROC-ready confidence scores

---

## 🧰 Requirements

| Dependency | Purpose |
|---|---|
| `open_clip_torch` | BiomedCLIP model loading |
| `segment-anything` | SAM ViT-H segmentation |
| `opencv-python-headless` | Image processing & connected components |
| `torch` + `torchvision` | Deep learning backend |
| `scipy` | Gaussian smoothing of attention maps |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` | Result visualization |
| `Pillow` | Image I/O |

Install all dependencies:

```bash
pip install open_clip_torch segment-anything opencv-python-headless
pip install git+https://github.com/openai/CLIP.git
```

---

## 🚀 Usage

### 1. Download the SAM Checkpoint

```python
import urllib.request

urllib.request.urlretrieve(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_h_4b8939.pth"
)
```

### 2. Load Models

```python
import open_clip
from segment_anything import sam_model_registry, SamPredictor

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL)
tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
clip_model.to(device).eval()

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
sam_predictor = SamPredictor(sam)
```

### 3. Run Zero-Shot Detection

```python
system = CLIPGuidedSAM(clip_model, clip_preprocess, tokenizer, sam_predictor, device)

ANOMALY_PROMPTS = {
    "COVID": [
        "COVID-19 infection, ground glass opacity, viral pneumonia",
        "patchy opacities, lung infection, inflammation",
        "lung consolidation, abnormal shadows, COVID infection"
    ],
    "Normal": ["healthy lungs, normal chest x-ray, no abnormalities"]
}

results = system.zero_shot_anomaly_detection(
    image_path="path/to/xray.png",
    anomaly_prompts=ANOMALY_PROMPTS["COVID"],
    normal_prompt=ANOMALY_PROMPTS["Normal"][0]
)

print(f"Confidence: {results['confidence']:.3f}")
print(f"Severity: {severity_from_confidence(results['confidence'])}")
```

### 4. Visualize Results

```python
visualize_results("path/to/xray.png", results)
```

The visualization renders a 4-panel figure: **Original → CLIP Attention → Differential Attention → SAM Mask overlay**.

---

## 📊 Output

Each detection call returns a dictionary:

| Key | Description |
|---|---|
| `confidence` | Final anomaly score in [0, 1] |
| `attention_map` | Differential attention (anomaly − normal) |
| `raw_attention_anomaly` | Raw CLIP gradient map for COVID prompts |
| `mask` | Binary segmentation mask from SAM |
| `points` | Candidate anomaly points fed to SAM |
| `similarity_anomaly` | Mean CLIP cosine similarity to COVID prompts |
| `sam_score` | Average SAM mask confidence score |

### Severity Thresholds

| Confidence | Severity |
|---|---|
| < 0.35 | Normal |
| 0.35 – 0.55 | Mild |
| 0.55 – 0.75 | Moderate |
| ≥ 0.75 | Severe |

### Evaluation Metrics

The notebook computes classification metrics on a sample of balanced images:

```
===== Zero-Shot CLIP+SAM Metrics =====
Accuracy : X.XXX
Recall   : X.XXX
F1-Score : X.XXX
```

---

## 📁 Dataset

The notebook is designed to work with a chest X-ray dataset structured as:

```
datasets/
├── COVID/
│   └── *.png
└── Normal/
    └── *.png
```

The dataset is fetched from Kaggle. Upload your `kaggle.json` API token when prompted in Google Colab.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `CONF_THRESHOLD` | `0.5` | Decision threshold for COVID vs Normal classification |
| `threshold` (attention) | `0.45` | Binarization threshold for anomaly region detection |
| `min_region_size` | `80` | Minimum pixel area for a valid anomaly region |
| `SAM_MODEL_TYPE` | `vit_h` | SAM model variant |
| Confidence blend | `0.85 / 0.15` | Weight of refined vs global CLIP score |

---

## 🔬 How It Works

1. **Attention map generation** — For each COVID text prompt, gradient backpropagation through CLIP produces a spatial attention map highlighting image regions most similar to the prompt.
2. **Differential masking** — The normal lung attention map is subtracted from the COVID attention map, suppressing healthy tissue and emphasizing pathological areas.
3. **Region detection** — Connected component analysis on the differential map identifies candidate anomaly coordinates above a spatial threshold.
4. **SAM segmentation** — These coordinates are used as point prompts for SAM, which produces a precise segmentation mask filtered by area constraints (0.1%–35% of image).
5. **Confidence refinement** — The masked region is re-encoded by CLIP against COVID prompts. This refined similarity is blended with a global image similarity score. Images with no detected mask have their confidence suppressed by 70%.

---

## 🛠️ Running in Google Colab

This notebook is optimized for Google Colab with GPU:

```
Runtime → Change runtime type → T4 GPU
```

All dependencies are installed in the first cell. The SAM checkpoint (~2.4 GB) is downloaded automatically if not already present.

---

## 📄 License

This project is for research and educational purposes. Model weights are subject to their respective licenses:
- [BiomedCLIP License](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [Segment Anything Model License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)

---

## 🙏 Acknowledgements

- [Microsoft BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — domain-adapted CLIP for biomedical images
- [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything) — universal image segmentation
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — open-source CLIP training and inference
