#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install git+https://github.com/facebookresearch/segment-anything.git')
get_ipython().system('pip install opencv-python numpy scikit-image matplotlib')


# In[71]:


# 1. First install required packages
import subprocess
import sys

def install_packages():
    packages = [
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "scikit-image",
        "matplotlib",
        "requests"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Install segment-anything from GitHub
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/facebookresearch/segment-anything.git"
    ])

# 2. Download SAM weights
def download_sam_weights():
    import requests
    import os
    
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    save_path = "sam_vit_b_01ec64.pth"
    
    if not os.path.exists(save_path):
        print("Downloading SAM weights...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print("Weights file already exists.")

# Run setup
if __name__ == "__main__":
    install_packages()
    download_sam_weights()
    print("\nSetup completed successfully!")
    print("You can now run the SAM training/evaluation script.")


# In[5]:


import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage import io, color
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.metrics import jaccard_score

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"  # Download from SAM repository

# Initialize SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)
predictor = SamPredictor(sam)


# In[94]:


import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
import os

# Configuration
DEVICE = "cpu"
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

# Initialize SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)


# In[95]:


import os
# Custom Dataset for histology images
class HistologyDataset(Dataset):
    def __getitem__(self, idx):
        # 1. Load and preprocess image
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 3. Load masks (replace with your actual mask loading logic)
        h, w = image.shape[:2]
        malignant_mask = np.zeros((h, w), dtype=np.uint8)
        non_malignant_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 4. Generate boxes from masks
        def get_boxes_from_mask(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return [[x, y, x+w, y+h] for (x,y,w,h) in 
                   [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]]
        
        malignant_boxes = get_boxes_from_mask(malignant_mask)
        non_malignant_boxes = get_boxes_from_mask(non_malignant_mask)

        return {
            'image': image_tensor,  # Shape: [3, H, W]
            'malignant_mask': torch.from_numpy(malignant_mask).long(),
            'non_malignant_mask': torch.from_numpy(non_malignant_mask).long(),
            'malignant_boxes': malignant_boxes,
            'non_malignant_boxes': non_malignant_boxes,
            'image_path': path
        }

def train_sam(train_images, num_epochs=3):
    # Initialize model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    # Create dataset
    dataset = HistologyDataset(train_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            try:
                # Get batch data - no need for image_tensor variable
                image_np = batch['image'][0].numpy().transpose(1, 2, 0)
                h, w = image_np.shape[:2]
                
                # Process with SAM
                predictor.set_image(image_np)
                boxes = batch['malignant_boxes'][0] + batch['non_malignant_boxes'][0]
                boxes_tensor = torch.tensor(boxes, device=DEVICE)
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, (h, w))
                
                # Get predictions
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
                
                # Calculate loss
                target = batch['malignant_mask'][0].to(DEVICE)
                loss = loss_fn(masks.squeeze(1), target)
                
                # Optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            except Exception as e:
                print(f"Error processing {batch['image_path'][0]}: {str(e)}")
                continue


# In[96]:


# Training function
def train_sam(train_images, num_epochs=6):
    print("Initializing SAM...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    dataset = HistologyDataset(train_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"\nStarting training on {len(dataset)} images...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        processed_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get image and convert to numpy (HWC format)
                image = batch['image'][0].numpy().transpose(1, 2, 0)
                h, w = image.shape[:2]
                
                # Get masks and boxes
                malignant_mask = batch['malignant_mask'][0].numpy()
                boxes = batch['malignant_boxes'][0] + batch['non_malignant_boxes'][0]
                
                if len(boxes) == 0:
                    print(f"⚠️ Batch {batch_idx} has no boxes - skipping")
                    continue
                
                # Process image with SAM
                predictor.set_image(image)
                boxes_tensor = torch.tensor(boxes, device=DEVICE)
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, (h, w))
                
                # Get predictions
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
                        points=None,
                        boxes=transformed_boxes,
                        masks=None
                    )
                    
                    low_res_masks, _ = predictor.model.mask_decoder(
                        image_embeddings=predictor.get_image_embedding(),
                        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False  # Single mask output
                    )
                
                # Resize masks to original image size
                masks = F.interpolate(
                    low_res_masks,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                
                # Prepare target and prediction tensors
                target = torch.tensor(malignant_mask, device=DEVICE).unsqueeze(0)  # Add batch dim
                pred = masks.squeeze(1)  # Shape: [1, H, W]
                
                # Verify shapes match
                if pred.shape[-2:] != target.shape[-2:]:
                    print(f"⚠️ Shape mismatch - pred: {pred.shape}, target: {target.shape}")
                    continue
                
                # Calculate loss
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                processed_batches += 1
                
                if (batch_idx + 1) % 2 == 0:
                    print(f"Batch {batch_idx+1} | Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Epoch summary
        if processed_batches > 0:
            avg_loss = epoch_loss / processed_batches
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - No valid batches processed")


# In[97]:


def evaluate_sam(test_images):
    dataset = HistologyDataset(test_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictor = SamPredictor(sam)
    iou_scores = []
    
    for batch in dataloader:
        try:
            image = batch['image'][0].numpy().transpose(1, 2, 0)  # Convert to HWC
            malignant_mask = batch['malignant_mask'][0].numpy()
            boxes = batch['malignant_boxes'][0]
            
            if len(boxes) == 0:
                print(f"No malignant boxes in {batch['image_path'][0]}")
                continue
                
            # Process image
            predictor.set_image(image)
            boxes_tensor = torch.tensor(boxes, device=DEVICE)
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, image.shape[:2])
            
            # Predict masks
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            
            # Combine masks
            pred_mask = masks[0][0].cpu().numpy()  # Take first mask
            
            # Calculate IoU
            iou = jaccard_score(
                malignant_mask.flatten() > 0,
                pred_mask.flatten() > 0,
                average='binary'
            )
            iou_scores.append(iou)
            
            # Visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(malignant_mask, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f"Predicted (IoU: {iou:.2f})")
            plt.axis('off')
            
            plt.show()
            
        except Exception as e:
            print(f"Error evaluating image: {str(e)}")
            continue
    
    if iou_scores:
        print(f"\nAverage IoU: {np.mean(iou_scores):.4f}")
    else:
        print("No valid evaluations completed")
    
    return iou_scores


# In[98]:


import os

# Test one file path
test_path = r"E:/Converted_Annotations/0a0bd016a9f4058f109f3b23ed9363bc11d69107_out.tif"
print(f"File exists: {os.path.exists(test_path)}")
print(f"File accessible: {os.access(test_path, os.R_OK)}")


# In[99]:


import os
from pathlib import Path

# Replace with your actual paths
PATHS_TO_CHECK = [
    r"E:/Converted_Annotations/00a3dc7e24407ce9673a3eacf38ab1cafa2a8a36_out.tif"
    
]

print("=== Path Verification ===")
print(f"Current working directory: {os.getcwd()}\n")

for path in PATHS_TO_CHECK:
    abs_path = Path(path).resolve()
    exists = os.path.exists(abs_path)
    
    print(f"Path: {abs_path}")
    print(f"Exists: {exists}")
    if exists:
        print(f"Readable: {os.access(abs_path, os.R_OK)}")
        try:
            with open(abs_path, 'rb') as f:
                print("OpenCV readable:", cv2.imread(str(abs_path)) is not None)
        except Exception as e:
            print(f"OpenCV test failed: {str(e)}")
    print("-" * 50)


# In[100]:


import os
import cv2
import numpy as np
from pathlib import Path

def verify_images(image_paths):
    """Check which images exist and are readable"""
    valid_images = []
    problematic = []
    
    for path in image_paths:
        try:
            # Convert to absolute path
            abs_path = str(Path(path).resolve())
            
            if not os.path.exists(abs_path):
                problematic.append(f"File not found: {abs_path}")
                continue
                
            img = cv2.imread(abs_path)
            if img is None:
                problematic.append(f"Unreadable image: {abs_path}")
                continue
                
            valid_images.append(abs_path)
            
        except Exception as e:
            problematic.append(f"Error checking {path}: {str(e)}")
    
    # Print results
    print(f"\nFound {len(valid_images)} valid images")
    if problematic:
        print("\nProblematic files:")
        for issue in problematic[:5]:  # Show first 5 problems
            print(f" - {issue}")
        if len(problematic) > 5:
            print(f" - ...and {len(problematic)-5} more")
    
    return valid_images

# Example usage:
if __name__ == "__main__":
    # Replace with your actual paths
    train_images = [
        r"E:/Converted_Annotations/0a0bd016a9f4058f109f3b23ed9363bc11d69107_out.tif",
        r"E:/Converted_Annotations/00a3dc7e24407ce9673a3eacf38ab1cafa2a8a36_out.tif",
        r"E:/Converted_Annotations/0a0ce1220f56a48bf14615f80bc4c684244c909d_out.tif",
        r"E:/Converted_Annotations/0a0d21d0131e154566005b3e5b6ad22e2262dfb7_out.tif",
        r"E:/Converted_Annotations/0a1bb809ebf076106fa9f329d01ea91ec6169a4d_out.tif",
        r"E:/Converted_Annotations/0a2ad317ba29981cb75808c76788b0af9fc6f12b_out.tif"
        # Add all your paths here
    ]
    
    print("Verifying training images...")
    valid_train_images = verify_images(train_images)
    
    if not valid_train_images:
        print("\n❌ Critical: No valid training images found!")
        print("Please check:")
        print("1. The paths are correct (copied from File Explorer)")
        print("2. The files exist at these locations")
        print("3. You have permission to access these files")
        print("4. The images are in supported formats (JPEG, PNG, TIFF)")
    else:
        print("\n✅ Successfully verified images")
        print("First valid image:", valid_train_images[0])


# In[101]:


# Example usage
if __name__ == "__main__":
    # =============================================
    # 1. Define your image paths (MODIFY THESE TO YOUR ACTUAL PATHS)
    # =============================================
    train_images = [
        r"E:/Converted_Annotations/00a3dc7e24407ce9673a3eacf38ab1cafa2a8a36_out.tif",
        r"E:/Converted_Annotations/0a0bd016a9f4058f109f3b23ed9363bc11d69107_out.tif",
        r"E:/Converted_Annotations/0a0ce1220f56a48bf14615f80bc4c684244c909d_out.tif",
        r"E:/Converted_Annotations/0a0d21d0131e154566005b3e5b6ad22e2262dfb7_out.tif",
        r"E:/Converted_Annotations/0a1bb809ebf076106fa9f329d01ea91ec6169a4d_out.tif",
        r"E:/Converted_Annotations/0a2ad317ba29981cb75808c76788b0af9fc6f12b_out.tif"
    ]
    
    test_images = [
        r"E:/Test_sample_images/0000ec92553fda4ce39889f9226ace43cae3364e_out.tif",
        r"E:/Test_sample_images/00a04c277c1a4bd14b7636d4c1c346d098a0f805_out.tif",
        r"E:/Test_sample_images/00a283d23e79001ac8e765f57f0894ab89d29c59_out.tif",
        r"E:/Test_sample_images/00b0834ff7de02164bc1d58b4bf052cbe568a926_out.tif",
        r"E:/Test_sample_images/00b3507842c0e0e8f11bad68feb5fb10089a2840_out.tif",
        r"E:/Test_sample_images/00d75d5647b3fb6ba1e2fc13c5ca5e5c2d0354cd_out.tif"
    ]

    # =============================================
    # 2. Verify paths before execution
    # =============================================
    def verify_paths(path_list, path_type):
        valid_paths = []
        print(f"\nVerifying {path_type} images:")
        for path in path_list:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                img = cv2.imread(abs_path)
                if img is not None:
                    valid_paths.append(abs_path)
                    print(f"✅ {abs_path}")
                else:
                    print(f"❌ Could not read (corrupted?): {abs_path}")
            else:
                print(f"❌ Not found: {abs_path}")
        return valid_paths

    train_images = verify_paths(train_images, "training")
    test_images = verify_paths(test_images, "test")

    if not train_images:
        print("\nERROR: No valid training images found!")
        exit()
    if not test_images:
        print("\nWARNING: No valid test images found!")

    # =============================================
    # 3. Execute training and evaluation
    # =============================================
    print("\nStarting training...")
    try:
        # Train the model
        train_sam(train_images, num_epochs=6)
        
        # Evaluate on test images if available
        if test_images:
            print("\nStarting evaluation...")
            iou_scores = evaluate_sam(test_images)
            print(f"\nFinal Average IoU: {np.mean(iou_scores):.4f}")
        else:
            print("\nSkipping evaluation (no valid test images)")
            
    except Exception as e:
        print(f"\nERROR during execution: {str(e)}")
        print("Debug info:")
        print(f"- Current working directory: {os.getcwd()}")
        print(f"- First train path: {train_images[0] if train_images else 'N/A'}")
        print(f"- First test path: {test_images[0] if test_images else 'N/A'}")


# In[ ]:





# In[ ]:


# Example usage
if __name__ == "__main__":
    # Paths to your training and test images
    train_images =[
        "E:/Converted_Annotations/00a3dc7e24407ce9673a3eacf38ab1cafa2a8a36_out.tif",
        "E:/Converted_Annotations/0a0bd016a9f4058f109f3b23ed9363bc11d69107_out.tif",
        "E:/Converted_Annotations/0a0ce1220f56a48bf14615f80bc4c684244c909d_out.tif",
        "E:/Converted_Annotations/0a0d21d0131e154566005b3e5b6ad22e2262dfb7_out.tif",
        "E:/Converted_Annotations/0a1bb809ebf076106fa9f329d01ea91ec6169a4d_out.tif",
        "E:/Converted_Annotations/0a2ad317ba29981cb75808c76788b0af9fc6f12b_out.tif"
    ]
    
    test_images =[
        "E:/Test_sample_images/0000ec92553fda4ce39889f9226ace43cae3364e_out.tif",
        "E:/Test_sample_images/00a04c277c1a4bd14b7636d4c1c346d098a0f805_out.tif",
        "E:/Test_sample_images/00a283d23e79001ac8e765f57f0894ab89d29c59_out.tif",
        "E:/Test_sample_images/00b0834ff7de02164bc1d58b4bf052cbe568a926_out.tif",
        "E:/Test_sample_images/00b3507842c0e0e8f11bad68feb5fb10089a2840_out.tif",
        "E:/Test_sample_images/00d75d5647b3fb6ba1e2fc13c5ca5e5c2d0354cd_out.tif"
    ]

 
   
 # Train the model
    print("Training SAM...")
    train_sam(train_images, num_epochs=6)
    
    # Evaluate on test images
    print("\nEvaluating SAM...")
    iou_scores = evaluate_sam(test_images)
    


# In[ ]:


def train_sam(train_images, num_epochs=6):
    print("Initializing SAM...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    
    # Initialize dataset with proper error handling
    try:
        dataset = HistologyDataset(train_images)
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty - no valid images found")
    except Exception as e:
        print(f"Dataset initialization failed: {str(e)}")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"Training on {len(dataset)} valid images...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        processed_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Safely get batch data
                if not all(k in batch for k in ['image', 'malignant_mask', 'non_malignant_mask']):
                    print(f"Batch {batch_idx} missing required keys")
                    continue
                
                image = batch['image'][0].numpy()  # Remove extra batch dimension
                malignant_mask = batch['malignant_mask'][0].numpy()
                non_malignant_mask = batch['non_malignant_mask'][0].numpy()
                
                # Create combined mask (0=background, 1=non-malignant, 2=malignant)
                combined_mask = np.zeros_like(malignant_mask)
                combined_mask[non_malignant_mask > 0] = 1
                combined_mask[malignant_mask > 0] = 2
                
                # Get boxes from masks
                boxes = batch['malignant_boxes'][0] + batch['non_malignant_boxes'][0]
                if len(boxes) == 0:
                    print(f"Batch {batch_idx} has no valid boxes")
                    continue
                
                # Convert to tensor
                boxes = torch.tensor(boxes, device=DEVICE)
                h, w = image.shape[:2]
                
                # Transform boxes
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes, (h, w))
                
                # Process image
                predictor.set_image(image)
                
                # Predict masks
                sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
                    points=None,
                    boxes=transformed_boxes,
                    masks=None,
                )
                
                low_res_masks, _ = predictor.model.mask_decoder(
                    image_embeddings=predictor.get_image_embedding(),
                    image_pe=predictor.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # Upscale masks
                masks = F.interpolate(
                    low_res_masks,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # Shape: (1, H, W)
                
                # Prepare target
                target = torch.tensor(combined_mask, device=DEVICE).long()
                
                # Calculate loss
                loss = loss_fn(masks, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                processed_batches += 1
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
        
        if processed_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/processed_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - No valid batches processed")

