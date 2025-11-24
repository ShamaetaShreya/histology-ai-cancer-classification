#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Paths to the dataset
train_dir = "E:/train"  # Update this path if needed
test_dir = "E:/test/test00"  # Update this path if needed
labels_csv = "E:/train_labels.csv"  # Update this path if needed

# Ensure 'image_path' contains full paths to images
labels_df['id'] = labels_df['id'] + '.tif'
labels_df['image_path'] = labels_df['id'].apply(lambda x: os.path.join(train_dir, x).replace("\\", "/"))

# Verify if image paths exist
labels_df['path_exists'] = labels_df['image_path'].apply(os.path.exists)
print(labels_df[['id', 'image_path', 'path_exists']].head(10))

# Filter only existing images
labels_df = labels_df[labels_df['path_exists']]
# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50




# In[7]:


# Data preprocessing and augmentation
def preprocess_data_gen_from_df(directory, labels_df, augment=False):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=augment,
        rotation_range=15 if augment else 0,
        width_shift_range=0.1 if augment else 0,
        height_shift_range=0.1 if augment else 0
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=directory,
        x_col='id',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=directory,
        x_col='id',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_gen, val_gen

train_gen, val_gen = preprocess_data_gen_from_df(train_dir, labels_df, augment=True)


# In[ ]:


# Build the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-10]:  # Fine-tune last 10 layers
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Load the test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# Predictions
test_preds = model.predict(test_gen, verbose=1)


# In[ ]:


# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen, verbose=1)
test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# Handling missing test labels safely
def get_test_labels(test_gen):
    filenames = test_gen.filenames
    labels = [labels_dict.get(os.path.splitext(os.path.basename(f))[0], '0') for f in filenames]
    return np.array(labels).astype(int)

test_labels = get_test_labels(test_gen)
predicted_labels = (test_preds > 0.5).astype(int)

# Confusion Matrix and Classification Report
cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

cr = classification_report(test_labels, predicted_labels, target_names=['Class 0', 'Class 1'])
print("Classification Report:")
print(cr)


# In[ ]:


# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

import tensorflow as tf
val_loss, val_accuracy = model.evaluate(val_gen, verbose=1)
test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Handling missing test labels safely
def get_test_labels(test_gen):
    filenames = test_gen.filenames
    labels = [labels_dict.get(os.path.splitext(os.path.basename(f))[0], '0') for f in filenames]
    return np.array(labels).astype(int)

test_labels = get_test_labels(test_gen)
predicted_labels = (test_preds > 0.5).astype(int)

# Confusion Matrix and Classification Report
cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

cr = classification_report(test_labels, predicted_labels, target_names=['Class 0', 'Class 1'])
print("Classification Report:")
print(cr)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot training and test loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(range(len(history.history['loss'])), [test_loss] * len(history.history['loss']), label='Test Loss', linestyle='dashed')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Save predictions, model, and training history
predictions_df = pd.DataFrame({"filename": test_gen.filenames, "prediction": np.squeeze(predicted_labels)})
predictions_df.to_csv("test_predictions.csv", index=False)

model.save("trained_model.h5")
np.save("training_history.npy", history.history)

print("Model training and evaluation complete. Predictions saved to test_predictions.csv. Model saved to trained_model.h5. Training history saved to training_history.npy.")


