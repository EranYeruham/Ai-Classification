from EasyCmm   import  visionLib as ec
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Setup train and testing paths
data_path = Path("data/")
image_path = data_path / "QA"
test_dir = image_path / "test"
imageSize = 64 

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor()
])


test_data_custom = ec.ImageFolderCustom(targ_dir=test_dir, 
                                    transform=test_transforms)
    

# Check out what's inside the training dataloader
class_names = test_data_custom.classes
#class_names = {"mouse","glasses","tool"}

# Need to setup model with input parameters
model_0 = ec.TinyVGG(  input_shape=3, # one for every pixel (64x64)
                            hidden_units=200, # how many units in the hiden
                            output_shape=len(class_names) # one for every c
                            ).to(device) # keep model on CPU to begin with 
    
#load saved model
model_0 = ec.LoadModel(model_0, device)

cap = cv2.VideoCapture(0)
ret,frame = cap.read()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,420)
frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)


transforms_image = test_transforms(frame)

    
# 5. Turn on model evaluation mode and inference mode
model_0.eval()
with torch.inference_mode():
    # Add an extra dimension to the image
    target_image = transforms_image.unsqueeze(dim=0)
    
    # Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model_0(target_image.to(device))
    print (target_image_pred)
    
# 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
# 7. Convert prediction probabilities -> prediction labels
target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

# 8. Plot the image alongside the prediction and prediction probability
plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
#plt.imshow(frame_rgb)
title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
                                  

plt.title(title)
plt.axis(False)
plt.show()
plt.show()
    