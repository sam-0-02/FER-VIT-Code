import os
import random
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

# Set source and destination directories
source_train_path = r"D:\Gaurab\Face restoration and emotion detection project\pythonProject\Fer 2013\train"
clean_save_path = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_clean\happy"
degraded_save_path = r"D:\Gaurab\Face restoration and emotion detection project\Step1_Output\train_degraded\happy"

# Create output folders if not exist
for folder in [clean_save_path, degraded_save_path]:
    for emotion in ['angry', 'happy', 'neutral', 'surprise']:
        os.makedirs(os.path.join(folder, emotion), exist_ok=True)


# Degradation Functions
def degrade_image(image: Image.Image):
    # Convert to numpy for noise addition
    np_img = np.array(image).astype(np.uint8)

    # Add Gaussian Noise
    noise = np.random.normal(0, 15, np_img.shape).astype(np.uint8)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_img)

    # Apply blur
    blurred_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Simulate JPEG compression by saving to buffer and reopening
    from io import BytesIO
    buffer = BytesIO()
    blurred_img.save(buffer, format="JPEG", quality=20)  # Low quality simulates compression
    buffer.seek(0)
    compressed_img = Image.open(buffer)

    return compressed_img


# Dataset loader (no transform because we'll handle it manually)
dataset = ImageFolder(root=source_train_path)

# Process and Save Images
print("Processing and saving clean + degraded images...")
for img, label in tqdm(dataset):
    class_name = dataset.classes[label]

    # Save original (clean)
    img_id = str(random.randint(100000, 999999)) + ".png"
    img.save(os.path.join(clean_save_path, class_name, img_id))

    # Save degraded version
    degraded_img = degrade_image(img)
    degraded_img.save(os.path.join(degraded_save_path, class_name, img_id))

print("✅ All images processed and saved.")
