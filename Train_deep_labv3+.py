
from Data_generator import CBISDDSMDataset as MammoDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from Data_generator import SegmentationTransform
from Data_generator import imshow, maskshow
import logging





# ________________________________
# 0. Load_model and logging
# ________________________________



logging.basicConfig(
    filename="training.log",   # Save logs in a file (or remove this line for console only)
    level=logging.INFO,        # Log INFO and above (can change to DEBUG for more detail)
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['val_loss']
        print(f"✅ Loaded checkpoint from epoch {start_epoch} with val_loss {best_loss:.4f}")
        return model, optimizer, start_epoch, best_loss
    else:
        print("⚠️ No checkpoint found, starting from scratch.")
        return model, optimizer, 0, float("inf")

# -----------------------
# 1. Get Model
# -----------------------
def get_deeplabv3(num_classes):
    model = deeplabv3_resnet50(pretrained=True)  # pretrained on COCO
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # output layer
    return model



# -----------------------
# 1.1 Get Model
# -----------------------
def get_deeplabv3_new(num_classes):
    # Load pretrained DeepLabv3+ with ResNet50 backbone
    model = deeplabv3_resnet50(pretrained=True)
    model = model.to(device)

    # Get original conv1 layer
    old_conv = model.backbone.conv1

    # Replace it with a new conv layer for 1-channel input
    model.backbone.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    # Copy pretrained weights (convert RGB weights to grayscale)
    with torch.no_grad():
        model.backbone.conv1.weight = nn.Parameter(
            old_conv.weight.sum(dim=1, keepdim=True)
        )

    # Update classifier for binary segmentation
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model




# -----------------------
# 4. Training Loop
# -----------------------
best_loss = float('inf')
def train_model(model, dataloader, optimizer, criterion, device):
    print(device)
    model.train()
    running_loss = 0.0
    counter = 1
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if counter % 5 == 0 : 
            print(f"the current loss after feeding {counter} is {running_loss / counter}")
            logging.info(f"the current loss after feeding {counter} is {running_loss / counter}"
                         )
        counter += 1 
    return running_loss / len(dataloader)

# -----------------------
# 5. Main
# -----------------------
if __name__ == "__main__":
    checkpoint_path = "checkpoint//"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired_transform = SegmentationTransform(img_size=(512, 512))

    train_dataset = MammoDataset(
    csv_path="final_datasets//training_dataset.csv",
    root_dir="",
    transform=paired_transform,
    img_size=(512, 512)
)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = get_deeplabv3_new(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    First_train = True
    if First_train:
        for epoch in range(num_epochs):
            loss = train_model(model, train_loader, optimizer, criterion, device)
            if loss < best_loss:
                best_loss = loss
                checkpoint_name = f"model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_loss
                    }, os.path.join("checkpoint", checkpoint_name))
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f}")
            
    else:
        checkpoint_path = "checkpoint/model_epoch_8.pth"  # change to whichever you want
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['val_loss']
        for epoch in range(checkpoint['epoch'] , 20):
            loss = train_model(model, train_loader, optimizer, criterion, device)
            if loss < best_loss:
                best_loss = loss
                checkpoint_name = f"model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_loss
                    }, os.path.join("checkpoint", checkpoint_name))
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f}")
            

    torch.save(model.state_dict(), "deeplabv3_mammo.pth")
