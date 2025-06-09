import cv2
import os
import glob

video_folder = 'videos/all_videos'  # .mkv videoların bulunduğu klasör
output_base_folder = 'frames'  # Karelerin kaydedileceği ana klasör
os.makedirs(output_base_folder, exist_ok=True)

# .mkv uzantılı tüm videoları bul
video_paths = glob.glob(os.path.join(video_folder, '*.mkv'))

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 2)  # 2 saniyede 1 kare
    frame_num = 0
    saved_count = 0

    print(f"İşleniyor: {video_name} | FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % interval == 0:
            filename = os.path.join(output_folder, f"{video_name}_frame_{frame_num}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_num += 1

    cap.release()
    print(f"{video_name} için {saved_count} kare kaydedildi.\n")
