

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Data_generator import CBISDDSMDataset as MammoDataset
from Data_generator import SegmentationTransform, imshow, maskshow, SegmentationTransform_test
from torchvision.models.segmentation import deeplabv3_resnet50
import os
import numpy as np


# -----------------------
# 1. Get Model Function
# -----------------------
def get_deeplabv3_new(num_classes, device):
    model = deeplabv3_resnet50(pretrained=True)
    model = model.to(device)

    # Fix conv1 for grayscale input
    old_conv = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    with torch.no_grad():
        model.backbone.conv1.weight = nn.Parameter(
            old_conv.weight.sum(dim=1, keepdim=True)
        )

    # Update classifier for num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# -----------------------
# 2. Load Model/Checkpoint
# -----------------------
def load_model(checkpoint_path, num_classes, device):
    model = get_deeplabv3_new(num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if checkpoint_path.endswith(".pth") and "checkpoint" in checkpoint_path:
        # Load checkpoint (resume training or eval)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
    else:
        # Load final trained model
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Loaded final trained model weights.")

    model.eval()
    return model


# -----------------------
# 3. Test/Evaluate Function
# -----------------------
def test_model(model, dataloader, device, num_images=5):
    model.eval()
    model = model.to(device)
    colors = {
    0: [0, 0, 0],        # black for background
    1: [0, 255, 0],      # green for normal
    2: [255, 0, 0]       # red for cancer
        }   
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)  # shape [B, H, W]

            # Show a few samples
            for i in range(min(len(images), num_images)):
                img_np = images[i].cpu().squeeze().numpy()
                mask_np = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()

                #convert to color for mask
                color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                for k, v in colors.items():
                    color_mask[mask_np == k] = v

                # convert to color predicted
                color_pred = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
                for k, v in colors.items():
                    color_pred[pred_np == k] = v

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np, cmap="gray")
                axs[0].set_title("Input Image")
                axs[1].imshow(color_mask, cmap="jet")
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(color_pred, cmap="jet")
                axs[2].set_title("Predicted Mask")
                plt.show()

            break  # only test one batch for visualization


# -----------------------
# 4. Run Test
# -----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (test/validation split)
    paired_transform = SegmentationTransform_test(img_size=(512, 512))
    test_dataset = MammoDataset(
        csv_path="final_datasets//test_dataset.csv",   # 👈 change to your test CSV
        root_dir="",
        transform=paired_transform,
        img_size=(512, 512)
    )
    batch_number = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_number, shuffle=False)

    # 👇 choose which to load
    #checkpoint_path = "checkpoint/model_epoch_8.pth"
    checkpoint_path = "checkpoint/model_epoch_19.pth"

    model = load_model(checkpoint_path, num_classes=3, device=device)

    # Run evaluation
    test_model(model, test_loader, device, batch_number)
