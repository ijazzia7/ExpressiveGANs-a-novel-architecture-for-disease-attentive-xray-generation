

    import torch
    import torch.nn as nn
    from torchvision import transforms, datasets
    from torchvision.datasets import CelebA
    from torch.utils.data import random_split, DataLoader
    import torch.optim as optim
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from tqdm.notebook import tqdm
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models


    def show_tensor_images_orig(image_tensor, num_images=25, size=(3, 64, 64), nrow=5, show=True):
        
        plt.figure(figsize=(8,8))
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        
        if show:
            plt.show()

    def set_requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def logits_map_to_prob_scalar(logits_map):
        """Average spatial logits -> sigmoid to get scalar per sample"""
        s = logits_map.mean(dim=[1,2,3])  # (B,)
        return torch.sigmoid(s)

    def show_image_tensor(img_tensors, num_images=5):
        """
        Show up to num_images images from a batch.
        img_tensors: (B,3,H,W) or (3,H,W) in [-1,1]
        """
        # If single image, add batch dimension
        if img_tensors.dim() == 3:
            img_tensors = img_tensors.unsqueeze(0)

        B = img_tensors.size(0)
        n = min(num_images, B)

        plt.figure(figsize=(n*3, 3))
        for i in range(n):
            im = img_tensors[i].detach().cpu().permute(1, 2, 0).numpy()
            if im.min() < 0:  # rescale from [-1,1] to [0,1]
                im = (im + 1.0) / 2.0
            plt.subplot(1, n, i+1)
            plt.imshow(im)
            plt.axis('off')
        plt.show()

    pip install -q torchxrayvision

Data Loader

    import torchxrayvision as xrv
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, datasets
    from PIL import Image
    import os

    # ----------- Transformations -----------
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # X-rays are single-channel
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # match Generator/Discriminator tanh output
    ])


    dataset = datasets.ImageFolder(
        root = '/kaggle/input/labeled-chest-xray-images/chest_xray',
        transform = transform
                        )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = next(iter(dataloader))
    print("Train batch:", images.shape, labels.shape)
    images = images.to(device)

Pretrained Teacher Architecture

    class CXR_CAM_Teacher(nn.Module):
        def __init__(self,meth = 'topk', model_name="densenet121-res224-all"):
            super().__init__()
            self.merg_meth = meth
            self.backbone = xrv.models.DenseNet(weights=model_name)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            for p in self.backbone.parameters():
                p.requires_grad = False

        def merge_cams(self, cams_up, logits=None, method="max", topk=3):
            B, C, H, W = cams_up.shape
        
            if method == "max":
                # Take the union (max) across channels
                merged = cams_up.max(dim=1, keepdim=True)[0]
        
            elif method == "mean":
                # Take the mean across channels
                merged = cams_up.mean(dim=1, keepdim=True)
        
            elif method == "weighted":
                assert logits is not None, "logits required for weighted method"
                probs = torch.sigmoid(logits)               # (B,C)
                probs = probs.unsqueeze(-1).unsqueeze(-1)   # (B,C,1,1)
                weighted = cams_up * probs
                merged = weighted.sum(dim=1, keepdim=True) / (probs.sum(dim=1, keepdim=True)+1e-6)

            elif method == "topk":
                assert logits is not None, "logits required for topk method"
                # Select top-k classes per sample
                top_idx = torch.topk(logits, k=topk, dim=1).indices  # (B, topk)
                merged_list = []
                for i in range(B):
                    selected = cams_up[i, top_idx[i]]  # (topk, H, W)
                    merged_list.append(selected.mean(dim=0, keepdim=True))  # (1,H,W)
                merged = torch.stack(merged_list, dim=0)  # (B,1,H,W)
        
            else:
                raise ValueError(f"Unknown method: {method}. Choose from ['max','mean','weighted','topk'].")
        
            # Normalize to [0,1] for visualization / stability
            merged = (merged - merged.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
            merged = merged / (merged.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        
            return merged

        def forward(self, x):
            with torch.no_grad():   # <---- critical line
                fmap = self.backbone.features(x)
                gap = self.pool(fmap).view(fmap.size(0), -1)
                logits = self.backbone.classifier(gap)

                W = self.backbone.classifier.weight
                fmap_perm = fmap.permute(0, 2, 3, 1)
                cams_map = torch.einsum('bhwc,kc->bhwk', fmap_perm, W)
                cams_map = cams_map.permute(0, 3, 1, 2)
                cams_up = F.interpolate(cams_map, size=(224, 224), mode="bilinear", align_corners=False)

                cams_out = self.merge_cams(cams_up,logits=logits, method=self.merg_meth, topk=3)

            return cams_out, cams_up, logits

    merged_union_t = CXR_CAM_Teacher(meth = 'max').to(device)
    merged_mean_t = CXR_CAM_Teacher(meth = 'mean').to(device)
    merged_weighted_t = CXR_CAM_Teacher(meth = 'weighted').to(device)
    merged_topk_t = CXR_CAM_Teacher(meth = 'topk').to(device)

    images = images.to(device)

    merged_union , _, _ = merged_union_t(images)
    merged_mean , _, _ = merged_mean_t(images)
    merged_weighted, _, _  = merged_weighted_t(images)
    merged_topk, _ , logits = merged_topk_t(images)

    merged_weighted.shape


    img_np = images[0].permute(1,2,0).cpu().numpy().squeeze()

    merged_maps = [
        ("Union (max)", merged_union[0]),
        ("Mean", merged_mean[0]),
        ("Weighted avg", merged_weighted[0]),
        (f"Top-k mean", merged_topk[0])
    ]

    # --- plot ---
    fig, axes = plt.subplots(1, len(merged_maps), figsize=(4*len(merged_maps), 4))
    if len(merged_maps) == 1:
        axes = [axes]

    for ax, (title, cam) in zip(axes, merged_maps):
        ax.imshow(img_np, cmap="gray")
        ax.imshow(cam.permute(1,2,0).detach().cpu().numpy(), cmap="jet", alpha=0.5)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    cams_up = _
    # pick the first image in batch
    img_np = images[0].permute(1,2,0).cpu().numpy().squeeze()

    cams_img = cams_up[0].cpu().detach().numpy()  # (18,224,224)
    probs = torch.softmax(logits[0], dim=0).cpu().detach().numpy()  # (18,)

    n_classes = cams_img.shape[0]
    cols = 6
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        if i < n_classes:
            ax.imshow(img_np, cmap="gray")
            ax.imshow(cams_img[i], cmap="jet", alpha=0.5)
            ax.set_title(f"Cls {i}\nP={probs[i]:.2f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

Discriminator Architecture

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DiscriminatorWithExplain(nn.Module):
        def __init__(self, in_channels=1, base_channels=64, cam_channels=1):
            super().__init__()
            
            # ----- Feature extractor -----
            self.conv1 = nn.Conv2d(in_channels, base_channels, 4, 2, 1)
            self.conv2 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)
            self.bn2 = nn.BatchNorm2d(base_channels*2)
            self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)
            self.bn3 = nn.BatchNorm2d(base_channels*4)
            self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1)
            self.bn4 = nn.BatchNorm2d(base_channels*8)
            self.conv5 = nn.Conv2d(base_channels*8, base_channels*8, 4, 2, 1)
            self.bn5 = nn.BatchNorm2d(base_channels*8)

            # ----- CAM head -----
            self.cam_head = nn.Conv2d(base_channels*8, cam_channels, kernel_size=1)

            # ----- Classification head -----
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(base_channels*8, 1)

        def forward(self, x):
            # convolutional backbone
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
            feat = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)  # (B, 512, H, W)

            # logits
            pooled = self.gap(feat).view(feat.size(0), -1)
            logits = self.fc(pooled)

            # cam map
            cam = torch.sigmoid(self.cam_head(feat))  # (B,1,H,W)
            cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)

            return logits, cam

Discriminator form old script

    class DiscriminatorWithExplain(nn.Module):
        def __init__(self):
            super(DiscriminatorWithExplain, self).__init__()
            self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.dense = nn.Linear(14*14*128, 1)
            self.dropout = nn.Dropout(0.4)
            self.flatten = nn.Flatten()
            
        def forward(self, x):
            x = self.dropout(F.leaky_relu(self.conv0(x), 0.2))
            x = self.dropout(F.leaky_relu(self.conv1(x), 0.2))
            x = self.dropout(F.leaky_relu(self.conv2(x), 0.2))
            x = self.dropout(F.leaky_relu(self.conv3(x), 0.2))
            x = self.flatten(x)
            x = torch.sigmoid(self.dense(x))
            return x, 1
            
            

Discriminator from old script (with cams)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DiscriminatorWithExplain(nn.Module):
        def __init__(self):
            super(DiscriminatorWithExplain, self).__init__()
            self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.dropout = nn.Dropout(0.4)
            self.flatten = nn.Flatten()
            self.dense = nn.Linear(14 * 14 * 128, 1)

        def forward(self, x):
            x = self.dropout(F.leaky_relu(self.conv0(x), 0.2))
            x = self.dropout(F.leaky_relu(self.conv1(x), 0.2))
            x = self.dropout(F.leaky_relu(self.conv2(x), 0.2))
            feat = self.dropout(F.leaky_relu(self.conv3(x), 0.2))  # last feature map (B,128,14,14)

            # Flatten for classification
            flat = self.flatten(feat)
            logits = self.dense(flat)  # (B,1)
            out = torch.sigmoid(logits)

            # ----------- Simple CAM computation -----------
            # Weight-based CAM (class activation map)
            w = self.dense.weight.view(1, 128, 14, 14)  # reshape dense weights
            cam = (feat * w.mean(dim=1, keepdim=True)).sum(dim=1, keepdim=True)  # (B,1,14,14)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam / (cam.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)  # normalize [0,1]

            return out, cam

    discriminator = DiscriminatorWithExplain().to(device)
    logits, cam = discriminator(images)

Generator

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GeneratorWithCAMGuidance(nn.Module):
        def __init__(self, z_dim=100, base_channels=512, out_channels=1):
            super().__init__()
            
            # ----- Core upsampling backbone (DCGAN-like) -----
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base_channels, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(True)
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(base_channels, base_channels//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//2),
                nn.ReLU(True)
            )
            self.up3 = nn.Sequential(
                nn.ConvTranspose2d(base_channels//2, base_channels//4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//4),
                nn.ReLU(True)
            )
            self.up4 = nn.Sequential(
                nn.ConvTranspose2d(base_channels//4, base_channels//8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//8),
                nn.ReLU(True)
            )
            self.up5 = nn.Sequential(
                nn.ConvTranspose2d(base_channels//8, base_channels//16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//16),
                nn.ReLU(True)
            )
            self.up6 = nn.Sequential(
                nn.ConvTranspose2d(base_channels//16, base_channels//32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//32),
                nn.ReLU(True)
            )
            self.up7 = nn.Sequential(
                nn.ConvTranspose2d(base_channels//32, base_channels//64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels//64),
                nn.ReLU(True)
            )
            
            # ----- Final Image Output -----
            self.final = nn.ConvTranspose2d(base_channels//64, out_channels, 3, stride=1, padding=1)
            
            # ----- Attention / CAM Heads -----
            self.attn1 = nn.Conv2d(base_channels, 1, kernel_size=1)
            self.attn2 = nn.Conv2d(base_channels//2, 1, kernel_size=1)
            self.attn3 = nn.Conv2d(base_channels//4, 1, kernel_size=1)
            self.attn4 = nn.Conv2d(base_channels//8, 1, kernel_size=1)
            self.attn5 = nn.Conv2d(base_channels//16, 1, kernel_size=1)
            self.attn6 = nn.Conv2d(base_channels//32, 1, kernel_size=1)
            self.attn7 = nn.Conv2d(base_channels//64, 1, kernel_size=1)

        def forward(self, z):
            attn_maps = []
            x = z.view(z.size(0), z.size(1), 1, 1)
            
            # progressive upsampling
            x = self.up1(x); attn_maps.append(torch.sigmoid(self.attn1(x)))
            x = self.up2(x); attn_maps.append(torch.sigmoid(self.attn2(x)))
            x = self.up3(x); attn_maps.append(torch.sigmoid(self.attn3(x)))
            x = self.up4(x); attn_maps.append(torch.sigmoid(self.attn4(x)))
            x = self.up5(x); attn_maps.append(torch.sigmoid(self.attn5(x)))
            x = self.up6(x); attn_maps.append(torch.sigmoid(self.attn6(x)))
            x = self.up7(x); attn_maps.append(torch.sigmoid(self.attn7(x)))
            
            # final image
            out = torch.tanh(self.final(x))  # [-1, 1] range
            out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
            
            return out, attn_maps

Generator 2

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class UpBlock(nn.Module):
        """Upsample + Conv2d (no checkerboard artifacts)."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.block(x)


    class GeneratorWithCAMGuidance(nn.Module):
        def __init__(self, z_dim=100, base_channels=512, out_channels=1):
            super().__init__()

            # ---- Initial projection ----
            self.fc = nn.Sequential(
                nn.Linear(z_dim, base_channels * 4 * 4),
                nn.ReLU(True)
            )

            # ---- Progressive upsampling ----
            self.up1 = UpBlock(base_channels, base_channels // 2)    # 4 → 8
            self.up2 = UpBlock(base_channels // 2, base_channels // 4)  # 8 → 16
            self.up3 = UpBlock(base_channels // 4, base_channels // 8)  # 16 → 32
            self.up4 = UpBlock(base_channels // 8, base_channels // 16) # 32 → 64
            self.up5 = UpBlock(base_channels // 16, base_channels // 32) # 64 → 128
            self.up6 = UpBlock(base_channels // 32, base_channels // 64) # 128 → 256
            self.up7 = UpBlock(base_channels // 64, base_channels // 128) # 256 → 512

            # ---- Final output conv ----
            self.final = nn.Sequential(
                nn.Conv2d(base_channels // 128, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

            # ---- Attention / CAM heads ----
            self.attn1 = nn.Conv2d(base_channels // 2, 1, kernel_size=1)
            self.attn2 = nn.Conv2d(base_channels // 4, 1, kernel_size=1)
            self.attn3 = nn.Conv2d(base_channels // 8, 1, kernel_size=1)
            self.attn4 = nn.Conv2d(base_channels // 16, 1, kernel_size=1)
            self.attn5 = nn.Conv2d(base_channels // 32, 1, kernel_size=1)
            self.attn6 = nn.Conv2d(base_channels // 64, 1, kernel_size=1)
            self.attn7 = nn.Conv2d(base_channels // 128, 1, kernel_size=1)


        def forward(self, z):
            B = z.size(0)
            x = self.fc(z).view(B, -1, 4, 4)

            attn_maps = []

            # progressive upsampling
            x = self.up1(x); attn_maps.append(torch.sigmoid(self.attn1(x)))
            x = self.up2(x); attn_maps.append(torch.sigmoid(self.attn2(x)))
            x = self.up3(x); attn_maps.append(torch.sigmoid(self.attn3(x)))
            x = self.up4(x); attn_maps.append(torch.sigmoid(self.attn4(x)))
            x = self.up5(x); attn_maps.append(torch.sigmoid(self.attn5(x)))
            x = self.up6(x); attn_maps.append(torch.sigmoid(self.attn6(x)))
            x = self.up7(x); attn_maps.append(torch.sigmoid(self.attn7(x)))

            out = self.final(x)
            out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)

            return out, attn_maps

# Generator from old script

    class GeneratorWithCAMGuidance(nn.Module):
        def __init__(self, z_dim=100):
            super(GeneratorWithCAMGuidance, self).__init__()
            self.dense = nn.Linear(z_dim, 4*4*256)
            self.bn0 = nn.BatchNorm1d(4*4*256)
            self.conv_t0 = nn.ConvTranspose2d(256, 256, kernel_size=2,
                                             stride=2, padding=1,
                                             output_padding=1)
            self.bn1 = nn.BatchNorm2d(256)
            self.conv_t1 = nn.ConvTranspose2d(256, 128, kernel_size=3,
                                             stride=2, padding=1,
                                             output_padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv_t2 = nn.ConvTranspose2d(128, 64, kernel_size=3,
                                             stride=2, padding=1,
                                             output_padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv_t3 = nn.ConvTranspose2d(64, 32, kernel_size=3,
                                             stride=2, padding=1,
                                             output_padding=1)
            self.bn4 = nn.BatchNorm2d(32)
            self.conv_t4 = nn.ConvTranspose2d(32, 16, kernel_size=3,
                                             stride=2, padding=1,
                                             output_padding=1)
            self.bn5 = nn.BatchNorm2d(16)
            self.conv_t5 = nn.ConvTranspose2d(16, 1, kernel_size=3,
                                             stride=2, padding=1,
                                             output_padding=1)

            
        def forward(self, x):
            x = self.dense(x)
            x = F.selu(self.bn0(x))
            x = x.view(-1, 256, 4, 4)
            x = F.selu(self.bn1(self.conv_t0(x)))
            x = F.selu(self.bn2(self.conv_t1(x)))
            x = F.selu(self.bn3(self.conv_t2(x)))
            x = F.selu(self.bn4(self.conv_t3(x)))
            x = F.selu(self.bn5(self.conv_t4(x)))
            x = F.tanh(self.conv_t5(x))
            return x, 1
        
        

            

Generator from old script but without checkerboard

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GeneratorWithCAMGuidance(nn.Module):
        def __init__(self, z_dim=100):
            super(GeneratorWithCAMGuidance, self).__init__()
            
            # Dense projection
            self.dense = nn.Linear(z_dim, 4 * 4 * 256)
            self.bn0 = nn.BatchNorm1d(4 * 4 * 256)

            # Each upsampling block: Upsample → Conv → BN → SELU
            def up_block(in_ch, out_ch):
                return nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.SELU(inplace=True)
                )

            self.up1 = up_block(256, 256)
            self.up2 = up_block(256, 128)
            self.up3 = up_block(128, 64)
            self.up4 = up_block(64, 32)
            self.up5 = up_block(32, 16)

            # Final conv to output 1-channel image
            self.final_conv = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            # Project latent vector
            x = self.dense(x)
            x = F.selu(self.bn0(x))
            x = x.view(-1, 256, 4, 4)

            # Progressive upsampling
            x = self.up1(x)  # 8x8
            x = self.up2(x)  # 16x16
            x = self.up3(x)  # 32x32
            x = self.up4(x)  # 64x64
            x = self.up5(x)  # 128x128

            # Final image
            x = torch.tanh(self.final_conv(x))  # output in [-1,1]
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # up to 224x224

            return x, 1

    z = torch.randn(images.shape[0], 100, device=device)
    generator = GeneratorWithCAMGuidance(z_dim=100).to(device)
    fake_imgs, _ = generator(z)

   
