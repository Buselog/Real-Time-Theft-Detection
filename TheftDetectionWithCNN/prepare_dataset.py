import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_train_test_split(original_folder='data', augmented_folder='data_augmented', combined_folder='data_combined', split_ratio=0.2):
    train_folder = os.path.join(combined_folder, 'train')
    test_folder = os.path.join(combined_folder, 'test')


    for category in ['theft', 'normal']:
        os.makedirs(os.path.join(train_folder, category), exist_ok=True)
        os.makedirs(os.path.join(test_folder, category), exist_ok=True)

        original_imgs = [os.path.join(original_folder, category, img) for img in os.listdir(os.path.join(original_folder, category))]
        augmented_imgs = [os.path.join(augmented_folder, category, img) for img in os.listdir(os.path.join(augmented_folder, category))]
        all_imgs = original_imgs + augmented_imgs

        train_imgs, test_imgs = train_test_split(all_imgs, test_size=split_ratio, random_state=42)

        for img_path in train_imgs:
            shutil.copy(img_path, os.path.join(train_folder, category, os.path.basename(img_path)))
        for img_path in test_imgs:
            shutil.copy(img_path, os.path.join(test_folder, category, os.path.basename(img_path)))


if __name__ == "__main__":
    prepare_train_test_split()
