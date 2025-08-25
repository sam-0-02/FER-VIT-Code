

import os
import cv2
import torch
import numpy as np
from gfpgan import GFPGANer
from PIL import Image
import matplotlib.pyplot as plt

# ====== Configurations ======
input_image_path = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_degraded\sample\istockphoto-1199509645-1024x1024.jpg"
model_path = r"D:\Gaurab\Face restoration and emotion detection project\Without perplexity on my own\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"
output_path = r"D:\Gaurab\Face restoration and emotion detection project\Step3_Output\restored_image.jpeg"

# ====== Step 1: Select Torch Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using Torch Device: {device} (GPU {'enabled' if device.type == 'cuda' else 'disabled'})")

# ====== Step 2: Read Input Image ======
print(f"📁 Checking input path: {input_image_path}")
img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"❌ Could not read image from: {input_image_path}")
print(f"✅ Image loaded: shape={img.shape}, dtype={img.dtype}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"🔄 Converted image from BGR to RGB")

# ====== Step 3: Initialize GFPGANer ======
print(f"📁 Checking model path: {model_path}")
restorer = GFPGANer(
    model_path=model_path,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device=device.type
)
print("✅ GFPGANer initialized successfully")

# ====== Step 4: Restore Face ======
print("🚀 Attempting face restoration...")
cropped_faces, restored_faces, restored_img = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

# Debug prints
print("✅ Face restoration completed")
print(f"🔍 cropped_faces: {len(cropped_faces)}")
print(f"🔍 restored_faces: {len(restored_faces)}")
print(f"🔍 restored_img_list: {type(restored_img)}")

if isinstance(restored_img, np.ndarray):
    print(f"🔎 restored_img shape: {restored_img.shape}, dtype: {restored_img.dtype}")

#-------Extra added-----------------------------------------------------------------------------------------------------
    def show_restoration_comparison(degraded_img, restored_img, figsize=(15, 5)):
        # Convert to numpy arrays if PIL Image
        if isinstance(degraded_img, Image.Image):
            degraded = np.array(degraded_img)
        else:
            degraded = degraded_img
        if isinstance(restored_img, Image.Image):
            restored = np.array(restored_img)
        else:
            restored = restored_img

        assert degraded.shape == restored.shape, "Degraded and restored images must be same shape!"

        diff = np.abs(restored.astype(np.float32) - degraded.astype(np.float32))
        if diff.ndim == 3:
            diff_gray = diff.mean(axis=2)
        else:
            diff_gray = diff

        diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.ptp() + 1e-8)

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        axs[0].imshow(degraded)
        axs[0].set_title("Degraded (Blurred) Image")
        axs[0].axis('off')

        axs[1].imshow(restored)
        axs[1].set_title("Restored Image")
        axs[1].axis('off')

        axs[2].imshow(restored)
        axs[2].imshow(diff_norm, cmap='jet', alpha=0.6)
        axs[2].set_title("Restoration Change Map")
        axs[2].axis('off')
        plt.tight_layout()
        plt.show()


    # ---- Place this line here ----
    show_restoration_comparison(img, restored_img)
    #------------------------------------Extra added--------------------------------------------------------------------------------
    # Convert back to BGR for saving with OpenCV
    restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, restored_bgr)
    print(f"💾 Restored image saved to: {output_path}")
else:
    raise ValueError("❌ Restored image is not a valid ndarray")

# -----------------------------------------------------------100 images restored  from each class-------------------------------
# import os
# import cv2
# import torch
# import numpy as np
# from gfpgan import GFPGANer
#
# def restore_images_in_folders(input_base_dir, output_base_dir, model_path, device_str='cuda'):
#     restorer = GFPGANer(
#         model_path=model_path,
#         upscale=1,
#         arch='clean',
#         channel_multiplier=2,
#         bg_upsampler=None,
#         device=device_str
#     )
#
#     # Identify emotion/class subfolders (e.g., angry, happy, etc.)
#     classes = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
#
#     for emotion in classes:
#         input_folder = os.path.join(input_base_dir, emotion)
#         output_folder = os.path.join(output_base_dir, emotion)
#         os.makedirs(output_folder, exist_ok=True)
#
#         print(f"\nProcessing emotion folder: {emotion}")
#         img_files = [f for f in os.listdir(input_folder)
#                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#         for img_file in img_files:
#             input_img_path = os.path.join(input_folder, img_file)
#             output_img_path = os.path.join(output_folder, img_file)
#
#             img = cv2.imread(input_img_path)
#             if img is None:
#                 print(f"Failed to read image: {input_img_path}, skipping.")
#                 continue
#
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             cropped_faces, restored_faces, restored_img = restorer.enhance(img_rgb, has_aligned=False, only_center_face=False, paste_back=True)
#
#             if not isinstance(restored_img, np.ndarray):
#                 print(f"Restored image invalid for: {input_img_path}, skipping.")
#                 continue
#
#             restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(output_img_path, restored_bgr)
#             print(f"Saved restored image: {output_img_path}")
#
# # ---- USER: UPDATE THESE PATHS ----
# input_degraded_base = r"D:\Gaurab\Face restoration and emotion detection project\Step3_Output\degraded"
# output_restored_base = r"D:\Gaurab\Face restoration and emotion detection project\Step3_Output\train_restored"
# model_path = r"D:\Gaurab\Face restoration and emotion detection project\Without perplexity on my own\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# restore_images_in_folders(input_degraded_base, output_restored_base, model_path, device)
#
