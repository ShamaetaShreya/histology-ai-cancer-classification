#!/usr/bin/env python
# coding: utf-8

# In[5]:


import subprocess
subprocess.run(["pip", "install", "numpy<2"])


# In[ ]:


# Activate the virtual environment (Windows specific)
import os
os.system("myenv\\Scripts\\activate")


# In[6]:


# Upgrade essential libraries
get_ipython().system('pip install --upgrade pandas')
get_ipython().system('pip install --upgrade scikit-learn')


# In[7]:


# Import necessary libraries
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[8]:


# Paths to dataset
train_dir = "E:/train"
test_dir = "E:/test/test00"
csv_file = "E:/train_labels.csv"

# Load the CSV file
labels_df = pd.read_csv(csv_file)

# Check the first few rows of the CSV
print(labels_df.head())


# In[9]:


# Add the full path to images
labels_df['id'] = labels_df['id'] + '.tif'
labels_df['image_path'] = labels_df['id'].apply(lambda x: os.path.join(train_dir, x).replace("\\", "/"))

# Verify if image paths exist
labels_df['path_exists'] = labels_df['image_path'].apply(os.path.exists)
print(labels_df[['id', 'image_path', 'path_exists']].head(10))


# In[13]:


# Define data generators
def define_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    return train_datagen, test_datagen

# Initialize the data generators
train_datagen, test_datagen = define_data_generators()



# In[14]:


# Convert labels to string (if necessary for the generator)
labels_df['label'] = labels_df['label'].astype(str)

# Create training and validation generators
train_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)


# In[15]:


# Create test generator
test_generator = test_datagen.flow_from_dataframe(
    labels_df,                # DataFrame containing the test images and their labels
    x_col='image_path',       # Column containing the path to images
    y_col='label',            # Column containing labels
    target_size=(224, 224),   # Resize the images to 224x224
    batch_size=32,
    class_mode='binary',
    shuffle=False
)


# In[16]:


# Load pre-trained InceptionV3 model without top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[17]:


# Build the model on top of InceptionV3
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Add the global average pooling layer
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
])


# In[18]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Setup EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    verbose=1,
    validation_data=test_generator,
    callbacks=[early_stopping]
)


# In[ ]:


# Evaluate the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Save the trained model
model.save('inceptionv3_trained_model.h5')

# Optionally, save the training history
np.save('inceptionv3_training_history.npy', history.history)

# Plot training and validation performance
import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

