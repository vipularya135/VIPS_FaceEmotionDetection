import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
import json

TRAIN_DIR = 'C:\\Users\\DELL\\Desktop\\face-emotion-det\\face_emotion_detection\\images\\train'
TEST_DIR = 'C:\\Users\\DELL\\Desktop\\face-emotion-det\\face_emotion_detection\\images\\validation'

def create_dataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train_df = pd.DataFrame()
train_df['image'], train_df['label'] = create_dataframe(TRAIN_DIR)
test_df = pd.DataFrame()
test_df['image'], test_df['label'] = create_dataframe(TEST_DIR)

from keras.preprocessing import image

def extract_features(images, target_size=(48, 48)):
    features = []
    for image_path in tqdm(images):
        img = image.load_img(image_path, target_size=target_size, color_mode='grayscale')
        img = image.img_to_array(img)
        features.append(img)
    features = np.array(features)
    return features


train_features = extract_features(train_df['image'])
test_features = extract_features(test_df['image'])

x_train = train_features / 255.0
x_test = test_features / 255.0

le = LabelEncoder()
le.fit(train_df['label'])
y_train = le.transform(train_df['label'])
y_test = le.transform(test_df['label'])
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

def create_model():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def print_model_details(history, model_name):
    print(f"Model: {model_name}")
    print("Training Accuracy: ", max(history.history['accuracy']))
    print("Validation Accuracy: ", max(history.history['val_accuracy']))

# Training the first model on full data
model_full = create_model()
history_full = model_full.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
print_model_details(history_full, "Full Data Model")

# Save the full model
model_json = model_full.to_json()
with open("nn_full_data_model.json", "w") as json_file:
    json_file.write(model_json)
model_full.save("nn_full_data_model.h5")

# Assuming high-confidence samples have a confidence score > threshold (e.g., 0.7)
# Let's simulate confidence scores for illustration
np.random.seed(42)
confidence_scores = np.random.rand(len(y_train))

threshold_high = 0.5
threshold_low = 0.5

high_conf_indices = np.where(confidence_scores > threshold_high)[0]
low_conf_indices = np.where(confidence_scores < threshold_low)[0]

x_train_high_conf = x_train[high_conf_indices]
y_train_high_conf = y_train[high_conf_indices]

x_train_low_conf = x_train[low_conf_indices]
y_train_low_conf = y_train[low_conf_indices]

# Print the number of images for each label/class
def print_class_distribution(y_data, label_encoder):
    labels = label_encoder.inverse_transform(np.argmax(y_data, axis=1))
    distribution = pd.Series(labels).value_counts()
    print(distribution)

print("Class distribution in Full Data:")
print_class_distribution(y_train, le)
print("Class distribution in High Confidence Data:")
print_class_distribution(y_train_high_conf, le)
print("Class distribution in Low Confidence Data:")
print_class_distribution(y_train_low_conf, le)

# Training the second model on high-confidence data
model_high_conf = create_model()
history_high_conf = model_high_conf.fit(x_train_high_conf, y_train_high_conf, batch_size=128, epochs=100, validation_data=(x_test, y_test))
print_model_details(history_high_conf, "High Confidence Data Model")

# Save the high confidence model
model_json = model_high_conf.to_json()
with open("high_conf_data_model.json", "w") as json_file:
    json_file.write(model_json)
model_high_conf.save("high_conf_data_model.h5")

# Training the third model on low-confidence data
model_low_conf = create_model()
history_low_conf = model_low_conf.fit(x_train_low_conf, y_train_low_conf, batch_size=128, epochs=100, validation_data=(x_test, y_test))
print_model_details(history_low_conf, "Low Confidence Data Model")

# Save the low confidence model
model_json = model_low_conf.to_json()
with open("low_conf_data_model.json", "w") as json_file:
    json_file.write(model_json)
model_low_conf.save("low_conf_data_model.h5")

