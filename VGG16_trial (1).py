#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install numpy<2


# In[3]:


pip list


# In[4]:


import os
os.system("myenv\\Scripts\\activate")  # For Windows



# In[4]:


where python


# In[6]:


get_ipython().system('pip install --upgrade pandas')
get_ipython().system('pip install --upgrade scikit-learn')


# In[8]:


import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split



# In[9]:


# Paths to dataset
train_dir = "E:/train"
test_dir = "E:/test/test00"
csv_file = "E:/train_labels.csv"

# Load the CSV file
labels_df = pd.read_csv(csv_file)

# Check the first few rows of the CSV
print(labels_df.head())


# In[10]:


import os

import pandas as pd

# Example: Construct the path to an image in the train dataset
image_name = labels_df.loc[0, 'id']  # Get the first image name from the DataFrame
image_path = os.path.join(train_dir, image_name)

print(f"Path to the first training image: {image_path}")


# In[11]:


# Print column names to verify
print(labels_df.columns)


# In[12]:


import os

# Construct the full path to each image
labels_df['image_path'] = labels_df['id'].apply(lambda x: os.path.join(train_dir, x).replace("\\", "/"))


# In[13]:


print(labels_df[['id', 'image_path']].head())


# In[14]:


labels_df['id'] = labels_df['id'] + '.tif'


# In[15]:


# Check if the images actually exist on the filesystem
labels_df['path_exists'] = labels_df['image_path'].apply(os.path.exists)

# Print the first 10 rows with path existence
print(labels_df[['id', 'image_path', 'path_exists']].head(10))


# In[16]:


import pandas as pd

#labels_df = pd.read_csv(csv_file)
for index, row in labels_df.iterrows():
    image_name = row['id']
    image_path = os.path.join(train_dir, image_name)

    print(f"Path to {image_name}: {image_path}")


# In[18]:


# Add the full path to images
labels_df['image_path'] = labels_df['id'].apply(lambda x: os.path.join(train_dir, x))


# In[35]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[36]:


# Data preprocessing and augmentation for training
#train_datagen = ImageDataGenerator(
   # rescale=1.0/255.0,
   # rotation_range=20,
   # width_shift_range=0.2,
   # height_shift_range=0.2,
   # shear_range=0.2,
   # zoom_range=0.2,
   # horizontal_flip=True
#)



# In[37]:


# Data preprocessing and augmentation for training data
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



# In[38]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Convert 'label' column to strings
labels_df['label'] = labels_df['label'].astype(str)

# Data Generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)


# Data generator for test images (only normalization, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Using flow_from_dataframe to prepare test data
test_generator = test_datagen.flow_from_dataframe(
    labels_df,                # DataFrame containing the test images and their labels
    x_col='image_path',        # Column containing the path to images
    y_col='label',             # Column containing labels
    target_size=(224, 224),    # Resize the images to 224x224
    batch_size=32,
    class_mode='binary',       # Binary classification
    shuffle=False              # Don't shuffle test data
)


# In[39]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# For test data, only rescale pixel values
#test_datagen = ImageDataGenerator(rescale=1./255)

# Convert label column to string
labels_df['label'] = labels_df['label'].astype(str)

# Data generator for training images
train_generator = train_datagen.flow_from_dataframe(
    labels_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)


# In[40]:


import os

# Check if the first 5 paths in the DataFrame actually exist
for image_path in labels_df['image_path'][:5]:
    if os.path.exists(image_path):
        print(f"Path exists: {image_path}")
    else:
        print(f"Path does NOT exist: {image_path}")


# In[42]:


# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model on top of VGG16
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Add the global average pooling layer
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Setup EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)



# In[ ]:


# Train the model
history = model.fit(
    train_generator,
    epochs=1,
    verbose=1
)


# In[1]:


# Evaluate the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")


# Save the trained model
model.save('vgg16_trained_model.h5')


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




