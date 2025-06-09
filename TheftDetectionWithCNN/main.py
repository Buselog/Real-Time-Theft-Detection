from utils.preprocess import extract_frames, augment_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from models.model import build_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import prepare_dataset

# Dataset augmentasyonu, hazırlık

original_theft_folder = 'data/theft'
original_normal_folder = 'data/normal'
augmented_theft_folder = 'data_augmented/theft'
augmented_normal_folder = 'data_augmented/normal'

# augment (veri çoğaltma)
augment_dataset(original_theft_folder, augmented_theft_folder, 'theft')
augment_dataset(original_normal_folder, augmented_normal_folder, 'normal')

# Train-test parçalama
prepare_dataset.prepare_train_test_split()

# Model Eğitimi

train_dir = 'data_combined/train'
test_dir = 'data_combined/test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

model = build_model()

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

model.save('theft_detection_model.h5')

# Test sonuçları yazdırma

test_generator.reset()
predictions = model.predict(test_generator)
pred_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.classes
print(classification_report(true_labels, pred_labels))
print(confusion_matrix(true_labels, pred_labels))

# Gerçek zamanlı kamera tespiti için kamera ayarları

model = load_model('theft_detection_model.h5')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

class_names = ['normal', 'theft']

frame_count = 0
predictions_buffer = []
label = "Bekleniyor..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame_rgb, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predictions_buffer.append(prediction[0])

        if len(predictions_buffer) > 5:
            predictions_buffer.pop(0)

        avg_pred = np.mean(predictions_buffer, axis=0)
        class_index = np.argmax(avg_pred)
        confidence = np.max(avg_pred)

        # Güven eşiği kontrolü
        if confidence < 0.75:
            label = "Bekleniyor..."
        else:
            label = class_names[class_index]

    text = "HIRSIZLIK YAPILDI" if label == 'theft' else ("NORMAL" if label == 'normal' else "Bekleniyor...")
    color = (0, 0, 255) if label == 'theft' else ((0, 255, 0) if label == 'normal' else (255, 255, 0))

    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Gercek Zamanli Hirsizlik Tespiti', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
