from EasyCmm   import  visionLib as ec
from pathlib import Path
from torch import nn
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer 
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#torch.manual_seed(42)

# Setup train and testing paths
data_path = Path("data/")
image_path = data_path / "QA"
train_dir = image_path / "train"
test_dir = image_path / "test" 
imageSize = 64 

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() ])
    
# Augment train data
train_transforms_simple = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor()
])


train_data_custom = ec.ImageFolderCustom(targ_dir=train_dir, 
                                    transform=train_transform_trivial_augment)

test_data_custom = ec.ImageFolderCustom(targ_dir=test_dir, 
                                    transform=test_transforms)
    
# Turn train and test custom Dataset's into DataLoader's
train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom c
                                    batch_size=1, # how many samples per batch
                                    num_workers = 0, # how many subprocesses t
                                    shuffle=True) # shuffle the data?
test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom cre
                                    batch_size=1, 
                                    num_workers = 0, 
                                    shuffle=False) # don't usually need to shu
    
    
class_names = train_data_custom.classes
#PlotDataLoaderImage(0,train_dataloader_custom)

# Check out what's inside the training dataloader
class_names = train_data_custom.classes

train_features_batch, train_labels_batch = next(iter(train_dataloader_custom))
print({train_features_batch.shape}, {train_labels_batch.shape})

# Need to setup model with input parameters
model_0 = ec.TinyVGG(  input_shape=3, # one for every pixel (64x64)
                            hidden_units=200, # how many units in the hiden
                            output_shape=len(class_names) # one for every c
                            ).to(device) # keep model on CPU to begin with 
    
#load saved model
model_0 = ec.LoadModel(model_0, device)
#pred_and_plot_image(model_0,"/home/eran/Programs/PyTorchVision/data/pizza_ste

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost functi
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)

start_time = timer() 

# Set the number of epochs (we'll keep this small for faster training times)
model_0_results = ec.train(model = model_0, 
                                train_dataloader = train_dataloader_custom, 
                                test_dataloader = test_dataloader_custom, 
                                optimizer = optimizer, 
                                loss_fn = loss_fn, 
                                device =device,
                                epochs = 10 )

ec.SaveModel(model_0)
ec.plot_loss_curves(model_0_results)

# Calculate training time      
end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")