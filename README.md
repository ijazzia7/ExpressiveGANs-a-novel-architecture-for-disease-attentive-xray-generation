# ExpressiveGAN‑CXR

Generating explainable synthetic chest X‑ray images with CAM‑driven GANs. A novel architecture for disease attentive xray generation

## Overview

In this project I developed a novel GAN architecture for medical imaging: a Generator guided by class‑activation maps (CAMs) and a Discriminator that not only distinguishes real from fake but also outputs attention heatmaps, enabling interpretability. My goal was to overcome the shortage of diverse, high‑quality chest X‑ray images (for example from the publicly available pneumonia‑CXR dataset) by synthetically expanding datasets, while preserving anatomical realism and diagnostic detail. A pretrained DenseNet teacher network extracts CAMs from real X‑rays, and these guide the Generator so it learns medically meaningful patterns rather than random textures. The DiscriminatorWithExplain produces both authenticity scores and attention heatmaps to make the system clinically transparent. While the focus is on pneumonia/normal chest X‑rays, the framework is broadly applicable to other safety‑critical domains (e.g., environmental monitoring, disaster assessment) where explainable synthetic image generation is needed.

## Dataset

- We use the publicly available pediatric chest X‑ray dataset (≈ 5,856 images) of children 1–5 years old from the Guangzhou Women and Children’s Medical Center in China. 

- BioMed Central

- The dataset includes labels: NORMAL / BACTERIA / VIRUS. 

- Images are split into training and testing sets of independent patients. 


## Dependencies include:

- PyTorch
- torchvision
- torchxrayvision
- matplotlib
- PIL (Pillow)
- numpy

## Usage
### 1. Prepare data

Modify the dataset path in dataset = datasets.ImageFolder(root=...) to point to your local copy of the chest X‑ray dataset.
Ensure transforms include grayscale conversion, resize to 224×224, normalization to match your Generator/Discriminator output range.

### 2. Pretrained teacher network

Instantiate the CXR_CAM_Teacher with merge method (max, mean, weighted, topk).
teacher = CXR_CAM_Teacher(meth='topk').to(device)
cams, ups, logits = teacher(images)

### 3. Define Generator and Discriminator

- GeneratorWithCAMGuidance takes a latent vector (z_dim = 100) and outputs a synthetic CXR plus attention maps at multiple layers.

- DiscriminatorWithExplain takes an image, outputs a logits score plus a CAM heatmap of shape (B,1,H,W).
Initialize weights with expressive_weights_init.

### 4. Training loop

- A simplified loop doubling as main training script:

- For each batch of real images: generate fake images.

- Train discriminator on real+fake, optionally add CAM‑loss term.

- Train generator to fool discriminator and optionally match teacher CAM maps.

- Log/generate example images, save checkpoint periodically.

### 5. Generating images

- After training, sample random z vectors, call generator(z) to get synthetic images and display them with show_image_tensor.

# NOTE: THIS IS AN ONGOING RESEARCH PROJECT
