# import os
# import random
# import torch
# import numpy as np
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
# from gfpgan import GFPGANer
# import cv2
#
# # Set paths
# clean_base = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_clean"
# degraded_base = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_degraded"
# gfpgan_model_path = r"D:\Gaurab\Face restoration and emotion detection project\Without perplexity on my own\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"
#
# # Emotion classes
# emotion_classes = ['angry', 'happy', 'surprise', 'neutral']
#
# # Device selection
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"✅ Using device: {device}")
#
# # Load Hugging Face ViT model
# model_name = "trpakov/vit-face-expression"
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
# model.eval()
# print(f"✅ Loaded model: {model_name}")
#
# # Load GFPGAN
# gfpganer = GFPGANer(
#     model_path=gfpgan_model_path,
#     upscale=1,
#     arch='clean',
#     channel_multiplier=2,
#     bg_upsampler=None,
#     device=device
# )
# print("✅ GFPGAN initialized")
#
# # Predict emotion from PIL image
# def emotion_predict(image: Image.Image):
#     try:
#         inputs = processor(images=image, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             pred_id = outputs.logits.argmax(-1).item()
#             return model.config.id2label[pred_id]
#     except Exception as e:
#         print(f"[❌] Error in prediction: {e}")
#         return "unknown"
#
# # Restore image using GFPGAN (from OpenCV image)
# def restore_image_cv2(img_path):
#     try:
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[❌] Could not read image: {img_path}")
#             return None
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         _, _, restored_img = gfpganer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
#         return Image.fromarray(restored_img)
#     except Exception as e:
#         print(f"[❌] GFPGAN error for {img_path}: {e}")
#         return None
#
# # Get up to 10 image paths per class
# def sample_image_paths(base_path):
#     paths = {}
#     for emotion in emotion_classes:
#         folder = os.path.join(base_path, emotion)
#         if not os.path.exists(folder):
#             print(f"[❌] Folder not found: {folder}")
#             continue
#         files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#         if not files:
#             print(f"[⚠️] No images in: {folder}")
#         paths[emotion] = random.sample(files, min(10, len(files)))
#     return paths
#
# # Sample image paths
# clean_paths = sample_image_paths(clean_base)
# degraded_paths = sample_image_paths(degraded_base)
#
# # Store results
# results = {
#     'clean': {'y_true': [], 'y_pred': []},
#     'degraded': {'y_true': [], 'y_pred': []},
#     'restored': {'y_true': [], 'y_pred': []}
# }
#
# # Predict on clean images
# for label, img_list in clean_paths.items():
#     for path in img_list:
#         try:
#             img = Image.open(path).convert("RGB")
#             pred = emotion_predict(img)
#             results['clean']['y_true'].append(label)
#             results['clean']['y_pred'].append(pred)
#         except Exception as e:
#             print(f"[❌] Error reading clean image: {path} | {e}")
#
# # Predict on degraded and restored images
# for label, img_list in degraded_paths.items():
#     for path in img_list:
#         try:
#             # Degraded image prediction
#             img = Image.open(path).convert("RGB")
#             pred_deg = emotion_predict(img)
#             results['degraded']['y_true'].append(label)
#             results['degraded']['y_pred'].append(pred_deg)
#
#             # Restored image prediction
#             restored = restore_image_cv2(path)
#             if restored:
#                 pred_res = emotion_predict(restored)
#                 results['restored']['y_true'].append(label)
#                 results['restored']['y_pred'].append(pred_res)
#             else:
#                 print(f"[⚠️] Skipped restoration for: {path}")
#         except Exception as e:
#             print(f"[❌] Error processing degraded image: {path} | {e}")
#
# # Print metrics
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
# import numpy as np
#
# def print_metrics(name, y_true, y_pred, emotion_classes):
#     print(f"\n=== {name.upper()} PERFORMANCE ===")
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     print(f"Accuracy  : {accuracy:.4f}")
#     print(f"Precision : {precision:.4f}")
#     print(f"Recall    : {recall:.4f}")
#
#     # Only print confusion matrix if all labels are valid
#     try:
#         cm = confusion_matrix(y_true, y_pred, labels=emotion_classes)
#         print("Confusion Matrix:")
#         print(cm)
#         print("Classification Report:")
#         print(classification_report(y_true, y_pred, target_names=emotion_classes, zero_division=0))
#     except ValueError as e:
#         print("⚠️ Error generating classification report or confusion matrix:")
#         print(str(e))
#
# # Output evaluation
# for key in results:
#     print_metrics(key, results[key]['y_true'], results[key]['y_pred'])


import os
import random
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from gfpgan import GFPGANer
import cv2

# Set paths
clean_base = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_clean"
degraded_base = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_degraded"
gfpgan_model_path = r"D:\Gaurab\Face restoration and emotion detection project\Without perplexity on my own\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"

# 7 classes in model, 4 target classes for evaluation
all_model_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
target_classes = ['angry', 'happy', 'neutral', 'surprised']  # Use 'surprised' to match model

# Map model output to your 4 classes (rest = "other" or "unknown")
def map_to_target_label(pred_label):
    label = pred_label.lower()
    if label in {'angry', 'happy', 'neutral', 'surprised'}:
        return label
    if label == 'surprise':  # just in case
        return 'surprised'
    return 'unknown'

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# Load model and processor
model_name = "trpakov/vit-face-expression"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
model.eval()
print(f"✅ Loaded model: {model_name}")

# GFPGAN load
gfpganer = GFPGANer(
    model_path=gfpgan_model_path,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device=device
)
print("✅ GFPGAN initialized")

def emotion_predict(image: Image.Image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = outputs.logits.argmax(-1).item()
            pred_label = model.config.id2label[pred_id]
            return map_to_target_label(pred_label)
    except Exception as e:
        print(f"[❌] Error in prediction: {e}")
        return "unknown"

def restore_image_cv2(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[❌] Could not read: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, _, restored_img = gfpganer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        return Image.fromarray(restored_img)
    except Exception as e:
        print(f"[❌] GFPGAN error for {img_path}: {e}")
        return None

def sample_image_paths(base_path):
    paths = {}
    for emotion in target_classes:
        folder = os.path.join(base_path, emotion)
        if not os.path.exists(folder):
            print(f"[❌] Folder not found: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"[⚠️] No images in: {folder}")
        paths[emotion] = random.sample(files, min(10, len(files)))
    return paths

# Sample
clean_paths = sample_image_paths(clean_base)
degraded_paths = sample_image_paths(degraded_base)

results = {
    'clean': {'y_true': [], 'y_pred': []},
    'degraded': {'y_true': [], 'y_pred': []},
    'restored': {'y_true': [], 'y_pred': []}
}

# Predict - CLEAN
for label, img_list in clean_paths.items():
    for path in img_list:
        try:
            img = Image.open(path).convert("RGB")
            pred = emotion_predict(img)
            results['clean']['y_true'].append(label)
            results['clean']['y_pred'].append(pred)
        except Exception as e:
            print(f"[❌] Error reading clean image: {path} | {e}")

# Predict - DEGRADED and RESTORED
for label, img_list in degraded_paths.items():
    for path in img_list:
        try:
            img = Image.open(path).convert("RGB")
            pred_deg = emotion_predict(img)
            results['degraded']['y_true'].append(label)
            results['degraded']['y_pred'].append(pred_deg)

            restored = restore_image_cv2(path)
            if restored:
                pred_res = emotion_predict(restored)
                results['restored']['y_true'].append(label)
                results['restored']['y_pred'].append(pred_res)
            else:
                print(f"[⚠️] Skipped restoration for: {path}")
        except Exception as e:
            print(f"[❌] Error processing degraded image: {path} | {e}")

def print_metrics(name, y_true, y_pred, classes):
    print(f"\n=== {name.upper()} PERFORMANCE ===")
    # Filter 'unknown' in predictions to focus metrics on real classes
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if p in classes:
            filtered_true.append(t)
            filtered_pred.append(p)
    if not filtered_true:
        print("[⚠️] No valid predictions in target classes!")
        return
    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision = precision_score(filtered_true, filtered_pred, average='macro', labels=classes, zero_division=0)
    recall = recall_score(filtered_true, filtered_pred, average='macro', labels=classes, zero_division=0)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    cm = confusion_matrix(filtered_true, filtered_pred, labels=classes)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(filtered_true, filtered_pred, labels=classes, target_names=classes, zero_division=0))

# Output
for key in results:
    print_metrics(key, results[key]['y_true'], results[key]['y_pred'], target_classes)
