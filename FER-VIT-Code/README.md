# FaceEmotionRestoration-ViT-GFPGAN

This repository contains the official code and resources for the paper:

**"Quality-aware facial emotion recognition using Vision Transformers with GFPGAN-based image restoration"**

---

##  Overview
Facial emotion recognition (FER) models are highly sensitive to image quality.  
This project proposes a **quality-aware pipeline** that:
1. Degrades face images synthetically to simulate real-world distortions (noise, blur, compression).  
2. Restores low-quality faces using **GFPGAN** (Generative Facial Prior GAN).  
3. Classifies emotions with a **Vision Transformer (ViT)** trained on the FER-2013 dataset.  
4. Evaluates the effect of restoration on classification accuracy and robustness.  

---

## 📂 Repository Structure
FaceEmotionRestoration-ViT-GFPGAN/
│── README.md <- Project documentation
│── LICENSE <- Open-source license (MIT)
│── requirements.txt <- Python dependencies
│── setup_instructions.md <- Detailed environment setup (optional)
│
│
├── data/ <- Dataset info & sample images
│ └── README.md <- How to download FER-2013
│
├── pretrained_models/ <- GFPGAN pretrained weights
│ └── GFPGANv1.3.pth
│ └── vit_model.pth
│
├── src/ <- Source code
│ ├── step1_degradation.py
│ ├── step2_quality_check.py
│ ├── step3_gfpgan_restore.py
│ ├── step4_vit_classification.py
│ ├── visualization.py
│ 
│
│
├── results/ <- Outputs from experiments
│ ├── figures/ <- ROC curves, confusion matrices, etc.
│ ├── tables/ <- CSV reports for metrics


---

## 📊 Dataset
We use the **FER-2013 dataset**, available at:  
👉 [FER-2013 Dataset (Kaggle)](https://www.kaggle.com/datasets/deadskull7/fer2013)

**Directory structure after download should be:**
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
   git clone https://github.com/yourusername/FaceEmotionRestoration-ViT-GFPGAN.git
   cd FaceEmotionRestoration-ViT-GFPGAN```

2. Create and activate a virtual environment:

	python -m venv env
	source env/bin/activate   # Linux/Mac
	env\Scripts\activate      # Windows

3. Install dependencies:
	pip install -r requirements.txt


## 🚀 Usage

Run each step sequentially:

Step 1 – Synthetic Degradation:
	
	python src/step1_degradation.py

Step 2 – Quality Check:
	python src/step2_quality_check.py

Step 3 – GFPGAN Restoration:
	python src/step3_gfpgan_restore.py --weights pretrained_models/GFPGANv1.3.pth

Step 4 – Emotion Classification (ViT):
	python src/step4_vit_classification.py

Visualization:
	python src/visualization.py


# 📈 Results

Image Quality Metrics: PSNR, SSIM

Emotion Classification Metrics: Accuracy, Precision, Recall, F1-score

Visualizations:

	Confusion matrices

	ROC curves

	Probability distributions

	Boxplots (contrast, sharpness, brightness, edge density)
	
# 📜 Citation

If you use this repository in your work, please cite:

Mudbhari G., Nandhini R., et al.
"Quality-aware facial emotion recognition using Vision Transformers with GFPGAN-based image restoration."
PeerJ Computer Science, 2025 (under review).

