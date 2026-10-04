#!/usr/bin/env python
# coding: utf-8

# In the name of God

# # Q2: Reconstruction of endoscopic polyp images with EndoVAE

# In this notebook we are going to be reproducing the work done in this paper (https://www.researchgate.net/publication/361927238_EndoVAE_Generating_Endoscopic_Images_with_a_Variational_Autoencoder) but we are going to sue teh Kvasir dataset from kaggle instead (https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)

# ## 2-1. Data preprocessing 

# For the first step we are going to preprocess the dataset by extracting the normal pictures from normal-z-line, normal-pylorus, normal-cecum folders and polyp images from polyp folders, change their color space to RGB, change their dimensions bilinearly to 96x96 and normalise them to [0, 1] range and perform some data augmentation to balance the datasets (we have 500 polyp images and 1500 normal images) and finally store them in a folder called preprocessed under the two classes of normal and polyp.
# 
# We will also visualise a few examples of images before and after this preprocessing.

# In[1]:


import os
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

dataset_path = './dataset' 
preprocessed_path = './processed'
image_size = 96
num_augmentation_per_polyp = 2

normal_folders = ['normal-cecum', 'normal-pylorus', 'normal-z-line']
polyp_folders = ['polyps'] 

polyp_augmentation_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
])

main_preprocess_transform = T.Compose([
    T.CenterCrop(480),
    T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR), 
    T.ToTensor()
])

output_normal_path = os.path.join(preprocessed_path, 'normal')
output_polyp_path = os.path.join(preprocessed_path, 'polyp')

os.makedirs(output_normal_path, exist_ok=True)
os.makedirs(output_polyp_path, exist_ok=True)

print("Processing normal images")
normal_image_paths = []
for folder in normal_folders:
    folder_path = os.path.join(dataset_path, folder)
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            normal_image_paths.append(os.path.join(folder_path, image_name))

for image_path in tqdm(normal_image_paths, desc="Normal Images"):
    try:
        image = Image.open(image_path).convert('RGB')
        processed_tensor = main_preprocess_transform(image)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = os.path.join(output_normal_path, f"{base_filename}.pt")
        torch.save(processed_tensor, output_filename)
    except Exception as e:
        print(f"Skipping file {image_path} due to error: {e}")

print("Processing polyp images")
polyp_image_paths = []
for folder in polyp_folders:
    folder_path = os.path.join(dataset_path, folder)
    for image_name in os.listdir(folder_path):
         if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            polyp_image_paths.append(os.path.join(folder_path, image_name))

total_polyps_processed = 0
for image_path in tqdm(polyp_image_paths, desc="Polyp Images"):
    try:
        image = Image.open(image_path).convert('RGB')
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        original_processed_tensor = main_preprocess_transform(image)
        output_filename_orig = os.path.join(output_polyp_path, f"{base_filename}.pt")
        torch.save(original_processed_tensor, output_filename_orig)
        total_polyps_processed += 1
        
        for i in range(num_augmentation_per_polyp):
            augmented_image = polyp_augmentation_transform(image)
            augmented_tensor = main_preprocess_transform(augmented_image)
            
            output_filename_aug = os.path.join(output_polyp_path, f"{base_filename}_aug_{i+1}.pt")
            torch.save(augmented_tensor, output_filename_aug)
            total_polyps_processed += 1
            
    except Exception as e:
        print(f"Skipping file {image_path} due to error: {e}")


# Now that the images are ready we will show a few of them

# In[2]:


import matplotlib.pyplot as plt
import random

polyp_folders = ['polyps']
output_polyp_path = os.path.join(preprocessed_path, 'polyp')

def show_before_after(original_image_path, processed_tensor_path):

    original_img = Image.open(original_image_path).convert('RGB')
    
    processed_tensor = torch.load(processed_tensor_path)
    processed_img = T.ToPILImage()(processed_tensor)
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Image: {os.path.basename(original_image_path)}", y=0.95)
    
    ax[0].imshow(original_img)
    ax[0].set_title("Before Preprocessing")
    ax[0].axis('off')
    
    ax[1].imshow(processed_img)
    ax[1].set_title("After Preprocessing (96x96)")
    ax[1].axis('off')
    
    plt.show()


polyp_folder_path = os.path.join(dataset_path, random.choice(polyp_folders))
all_original_polyp_images = [
    f for f in os.listdir(polyp_folder_path) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

if len(all_original_polyp_images) >= 5:
    selected_image_names = random.sample(all_original_polyp_images, 5)

    for img_name in selected_image_names:
        original_path = os.path.join(polyp_folder_path, img_name)
        processed_filename = os.path.splitext(img_name)[0] + ".pt"
        processed_path = os.path.join(output_polyp_path, processed_filename)
        
        show_before_after(original_path, processed_path)


# ## 2-2. EndoVAE architecture

# In this section we will explain the architecture of the EndoVAE as introduced in the paper and then implement it.
# 
# As with any other VAE, this model has three main components:
# 
# __1. The encoder__
# 
# The encoder compresses a high dimensional image (in our case 96x96x3) into a smaller set of features called the latent space. The six 2D convolution layers here act as feature extractors. As the image passes through them the dimension is reduced (when the stride is higher than 1) and this forces the model to learn abstract features. After this filtering the features are passed to a linear layer with 256 neurons for final preprocessing after which two separate linear layer, fc_mu and fc_logvar, map these features to the mean and log-variance of a gaussian distribution (which is the key difference between normal auto encoders and variational auto encoders)
# 
# __2. The reparameterization__
# 
# This is a clever trick that makes VAEs trainable. Since we are dealing with a random distribution rather than a fixed vector in VAEs we need to sample a point from this random distributions at this point, but random sampling has no gradient. The reparametrization trick solves this issue by separating the random part. We first sample a random number (vector) from a fixed standard normal distribution, which doesn't have any learnable parameters, then we scale this number (vector) by our learned standard deviation and shift it by our learned mean. This way the random part is external and the gradient can be calculated.
# 
# __3. The decoder__
# 
# The decoder's role is the reverse of the encoders, it takes a low dimensional vector from the latent space (form the previous layers) and attempts to reconstruct an image from it. It begins by using a linear layer to project the latent space into a larger tensor and create a base feature map from it then transpose convolution layers are used to slowly upsample this feature map and increase its size. Then another final convolution layer is used to condense all these feature maps into a 3 channel RGB image.
# 
# We will add all the layers according to this image from the paper and the section III of the paper where other details are motioned.
# 
# ![image.png](attachment:image.png)
# 
# One important note here is despite the fact that the diagram shows the usage of a mix of transpose convolution and normal convolution, the text of the paper states : "The decoder part of the proposed model consists of a fully connected layer that contains 36,864 neurons followed by seven transpose convolutional layers with 256, 128, 64, 64, 32, 32 and 3 filters respectively" so I implemented the decoder part using only transpose convolution.

# In[5]:


import torch.nn as nn

class EndoVAE(nn.Module):
    def __init__(self, latent_dim=6):
        super(EndoVAE, self).__init__()
        
        # Encoder 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_after_flatten = nn.Linear(256 * 12 * 12, 256)
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        
        # Decoder 
        self.decoder_input = nn.Linear(latent_dim, 256 * 12 * 12)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) 
        return mu + std * epsilon

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        encoded = self.fc_after_flatten(encoded)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder_input(z)
        decoded = decoded.view(-1, 256, 12, 12)
        reconstructed_x = self.decoder(decoded)
        
        return reconstructed_x, x, mu, logvar

model = EndoVAE(latent_dim=6)
print(model)


# ## 2-3. The cost functions

# In this section we are going to explain and implement the cost functions used in the paper.
# 
# For EndoVAE two losses were used:
# 
# __1. Reconstruction Loss (binary cross-entropy - BCE)__
# 
# The reconstruction loss measures how accurately the decoder can reconstruct the original image, after it has been compressed to the latent space. we use binary cross-entropy (calculated using the formula below) because the final layer in our decoder is a sigmoid. This scales every output pixel value to [0, 1] which can be interpreted as a probability.
# 
# ![image.png](attachment:image.png)
# 
# Why not MSE? While mean squared error (calculated using the formula below) could also work, BCE often works better for image reconstruction when the output pixel values are between 0 and 1 (in most cases I looked at they used BCE). This could be due to the fact that using BCE, it penalizes the model more strongly when the model is very confident and is wrong.
# 
# ![image-2.png](attachment:image-2.png)
# 
# __2. KL Divergence__
# 
# This is the part that makes a VAE a generative model. The Kullback-Leibler (KL) Divergence (calculated by the formula below) acts as a regularizer on the latent space. Its goal is to force the distributions learned by the encoder to be as close as possible to a standard normal distribution. Without it the encoder would learn to memorize the training data by placing each image in its own isolated point in the latent space. We want the latent space to resemble a gaussian distribution so we can later sample from this space and generate new images that look similar to the original dataset.
# 
# ![image-3.png](attachment:image-3.png)
# 
# These two losses are then combined in a __total loss__, which is calculated using the formula below, to strike a balance between preserving the information of the input (being a perfect autoencoder) and having a well organized latent space.
# 
# ![image-4.png](attachment:image-4.png)

# In[6]:


import torch.nn.functional as F

def vae_loss_function(reconstructed_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(
        reconstructed_x.view(-1, 3*96*96), 
        x.view(-1, 3*96*96), 
        reduction='sum'
    )
    
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kld
    
    return total_loss, recon_loss, kld


# In[ ]:




