# FER-2013 Dataset Information

This project uses the **FER-2013 dataset**, originally released in the  
Kaggle competition: ["Challenges in Representation Learning: Facial Expression Recognition Challenge"](https://www.kaggle.com/datasets/deadskull7/fer2013).

---

## ⚠️ Important Note
Due to licensing restrictions, the dataset **is not included** in this repository.  
Users must manually download it from Kaggle.

---

## 📥 Download Instructions
1. Go to the Kaggle dataset page:  
   👉 https://www.kaggle.com/datasets/deadskull7/fer2013  
2. Log in with your Kaggle account.  
3. Click **Download** to obtain `fer2013.zip`.  
4. Extract the contents of the zip file.  

---

## 📂 Expected Directory Structure
After extraction, arrange the dataset as follows:

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

## 🖼️ Sample Images
We provide a few **sample images** in `sample_images/` for testing the pipeline without downloading the full dataset.  

- `sample_images/clean/` → Example clean images  
- `sample_images/degraded/` → Example degraded images  
- `sample_images/restored/` → Example GFPGAN-restored images  

---

## 📜 Citation
If you use FER-2013, please cite the original authors:

Goodfellow, I., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013).
Challenges in Representation Learning: A report on three machine learning contests.
Neural Information Processing, 117-124.