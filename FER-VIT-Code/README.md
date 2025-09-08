# FaceEmotionRestoration-ViT-GFPGAN

This repository contains the official code and resources for the paper:

**"Quality-aware facial emotion recognition using Vision Transformers with GFPGAN-based image restoration"**

---

## 📌 Overview
Facial emotion recognition (FER) models are highly sensitive to image quality.  
This project proposes a **quality-aware pipeline** that:
1. Degrades face images synthetically to simulate real-world distortions (noise, blur, compression).  
2. Restores low-quality faces using **GFPGAN** (Generative Facial Prior GAN).  
3. Classifies emotions with a **Vision Transformer (ViT)** trained on the FER-2013 dataset.  
4. Evaluates the effect of restoration on classification accuracy and robustness.  

---

## 📂 Repository Structure
FER-VIT-Code/
│── README.md                       
│── LICENSE                         
│── requirements.txt                
│── setup_instructions.md           
│
├── data/
│   │── README.md                  
│   └── sample_images/              
│       ├── clean/
│       ├── degraded/
│       └── restored/
│
├── pretrained_models/
│   └── GFPGANv1.3.pth              
│
├── src/
│   ├── degradation and comparision step1.py
│   ├── cchecking for blur or clean image.py
│   ├── GFP GAN Step 3.py
│   ├── step 4 from restore to vit.py
│   ├── visualization.py
│



---

## 📊 Dataset
We use the **FER-2013 dataset**, available at:  
👉 [FER-2013 Dataset (Kaggle)](https://www.kaggle.com/datasets/deadskull7/fer2013)

**Important:** Due to licensing restrictions, this dataset **is not redistributed in this repository**.  
Users must download it from Kaggle and arrange the folders as follows:

data/FER-2013/
├── train/
│ ├── angry/
│ ├── happy/
│ ├── neutral/
│ └── surprise/
└── test/
├── angry/
├── happy/
├── neutral/
└── surprise/


---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/sam-0-02/FER-VIT-Code.git
   cd FER-VIT-Code
 ```

2. Create and activate a virtual environment:
```
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows
```

3. Install dependencies (tested with Python 3.10 and PyTorch 2.2.2):
```pip install -r requirements.txt```


## 💻 Code Information

The src/ folder contains all scripts required to reproduce the experiments:

src/
├── degradation and comparision step1.py      # Step 1: Apply synthetic degradation (Gaussian noise, blur, JPEG)
├── cchecking for blur or clean image.py      # Step 2: Quality assessment (detect blurred or clean images)
├── GFP GAN Step 3.py                         # Step 3: Restore degraded images using GFPGAN (v1.3 weights)
├── step 4 from restore to vit.py             # Step 4: Emotion classification using ViT (trpakov/vit-face-expression)
├── visualization.py                          # Generate ROC curves, confusion matrices, boxplots, etc.

Language: Python (tested with Python 3.10)

Frameworks: PyTorch, Hugging Face Transformers

Models used:

GFPGAN v1.3 (pretrained weights provided in pretrained_models/)

Vision Transformer (ViT) from Hugging Face (trpakov/vit-face-expression)

Execution order: Run step 1 → step 2 → step 3 → step 4 → visualization

##   🚀 Usage

Run each step sequentially:

Step 1 – Synthetic Degradation:
```` python src/step1_degradation.py```


Step 2 – Quality Check
```python src/step2_quality_check.py```

Step 3 – GFPGAN Restoration
```python src/step3_gfpgan_restore.py --weights pretrained_models/GFPGANv1.3.pth```

Step 4 – Emotion Classification (ViT)
```python src/step4_vit_classification.py```

Visualization
```python src/visualization.py```


## 🧪 Methodology (for reproducibility)

The pipeline follows these steps:

Degradation Simulation – Gaussian noise, Gaussian blur, and JPEG compression are applied to clean FER-2013 images to create degraded samples.

Frame Quality Assessment – Images are analyzed using Laplacian variance (blur) and Signal-to-Noise Ratio (SNR) to classify them as clean/problematic.

Restoration with GFPGAN – Degraded faces are restored using pretrained GFPGAN (v1.3) weights.

Emotion Classification – Both clean, degraded, and restored images are classified using a Vision Transformer model (trpakov/vit-face-expression).

Evaluation – Metrics include PSNR, SSIM (image quality) and Accuracy, Precision, Recall, F1-score (emotion classification). Visualizations include ROC curves, confusion matrices, and probability distributions.


## 📈 Results

Image Quality Metrics: PSNR, SSIM

Emotion Classification Metrics: Accuracy, Precision, Recall, F1-score

Visualizations: Confusion matrices, ROC curves, probability distributions, boxplots


## 📜 Citation

If you use this repository in your work, please cite:

```Mudvari, G., Nandhini, R., et al.
"Quality-aware facial emotion recognition using Vision Transformers with GFPGAN-based image restoration."
PeerJ Computer Science, 2025 (under review).```



## 🙏 Acknowledgements

FER-2013 dataset: Kaggle

GFPGAN: Tencent ARC Lab

ViT model: Hugging Face – trpakov/vit-face-expression


## 🤝 Contribution Guidelines

Contributions are welcome!
If you wish to improve this codebase, please open an issue or submit a pull request.
