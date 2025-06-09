import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, label, frame_rate=30, start_frame=0, end_frame=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = end_frame or total_frames

    frame_idx = 0
    saved_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break
        if start_frame <= frame_idx <= end_frame and frame_idx % frame_rate == 0:
            filename = f"{label}_{os.path.basename(video_path).split('.')[0]}_{saved_idx}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()


def augment_image(image):
    rows, cols = image.shape[:2]
    angle = np.random.choice([0, 90, 180, 270])
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    if np.random.rand() > 0.5:
        rotated = cv2.flip(rotated, 1)
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    brightness = np.random.uniform(0.8, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_img

def augment_dataset(input_folder, output_folder, label, augment_count=3):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(output_folder, f'{base_name}_orig.jpg'), image)
        for i in range(augment_count):
            aug_img = augment_image(image)
            new_name = f'{base_name}_aug{i+1}.jpg'
            cv2.imwrite(os.path.join(output_folder, new_name), aug_img)

