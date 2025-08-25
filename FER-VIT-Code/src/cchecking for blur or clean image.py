# import os
# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# # Path to clean and degraded images
# image_folder_path = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_degraded"
#
# # Set quality thresholds (tweak based on test observations)
# LAPLACIAN_BLUR_THRESHOLD = 100.0  # Lower = more blur
# SNR_THRESHOLD = 15.0  # Lower = more noise
#
#
# # Function to compute Laplacian variance (blur detector)
# def compute_blur_score(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return lap_var
#
#
# # Function to compute SNR (noise detector)
# def compute_snr(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mean = np.mean(gray)
#     stddev = np.std(gray)
#     snr = mean / (stddev + 1e-8)  # Avoid division by zero
#     return snr
#
#
# # Function to classify image
# def classify_quality(image):
#     blur_score = compute_blur_score(image)
#     snr_score = compute_snr(image)
#
#     is_clean = blur_score >= LAPLACIAN_BLUR_THRESHOLD and snr_score >= SNR_THRESHOLD
#
#     return is_clean, blur_score, snr_score
#
#
# # Run quality check on all images
# print("Assessing frame quality...")
# results = []
#
# for root, _, files in os.walk(image_folder_path):
#     for file in tqdm(files):
#         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             file_path = os.path.join(root, file)
#             image = cv2.imread(file_path)
#             is_clean, blur_val, snr_val = classify_quality(image)
#             results.append((file_path, is_clean, blur_val, snr_val))
#
# # Summary Report
# print("\nSample Output (First 10):")
# for entry in results[:10]:
#     status = "CLEAN" if entry[1] else "PROBLEMATIC"
#     print(f"{os.path.basename(entry[0])} → {status} | Blur: {entry[2]:.2f} | SNR: {entry[3]:.2f}")
#
# # Optional: Save classification result to file
# import csv
#
# output_csv = os.path.join(image_folder_path, "quality_report.csv")
# with open(output_csv, "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Filename", "Status", "Blur_Score", "SNR_Score"])
#     for file_path, is_clean, blur, snr in results:
#         writer.writerow([os.path.basename(file_path), "Clean" if is_clean else "Problematic", blur, snr])
#
# print(f"\n✅ Frame quality analysis completed and saved to: {output_csv}")


import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# ---------- Config ----------
parent_folder = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_clean"
emotions = ['angry', 'happy', 'surprise', 'neutral']
blur_threshold = 150.0  # Laplacian variance
snr_threshold = 10.0  # Signal to Noise Ratio
output_csv_path = os.path.join(parent_folder, "clean_folder_quality_report.csv")


# ----------------------------

def calculate_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def calculate_snr(image):
    mean = np.mean(image)
    std = np.std(image)
    return mean / (std + 1e-8)


results = []

print("🔍 Scanning Clean Image Folders...\n")

for emotion in emotions:
    folder_path = os.path.join(parent_folder, emotion)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in tqdm(image_files, desc=f"Processing {emotion}"):
        img_path = os.path.join(folder_path, image_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        blur_val = calculate_blur(image)
        snr_val = calculate_snr(image)

        label = "CLEAN" if (blur_val >= blur_threshold and snr_val >= snr_threshold) else "PROBLEMATIC"

        results.append({
            "Emotion": emotion,
            "Image": image_file,
            "Blur": round(blur_val, 2),
            "SNR": round(snr_val, 2),
            "Assessment": label
        })

# Save the results
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)

# Print summary
clean_count = df[df["Assessment"] == "CLEAN"].shape[0]
prob_count = df[df["Assessment"] == "PROBLEMATIC"].shape[0]

print(f"\n✅ Scan complete. Total Images: {len(df)}")
print(f"👍 Clean: {clean_count}")
print(f"⚠️ Problematic: {prob_count}")
print(f"\n📄 Detailed report saved to:\n{output_csv_path}")
