#!/usr/bin/env python
# coding: utf-8

# In the name of God

# # Q2: Robust Zero-Shot Classification

# ## 2-0 Introduction:

# In this notebook we are going to implement different ways of adversarial attacks and comparing them, we will be following to work done in this paper (https://openreview.net/forum?id=P4bXCawRi5J)

# ## 2-1. Introduction to the CLIP Model, Zero-Shot Classification, and Adversarial Attacks 

# ### 2-1-1. FGSM and PGD

# Before we explain what FGSM and PGD are let's first briefly explain what are adversarial attacks: 
# 
# __Adversarial attacks__: As the paper described, an adversarial attack adds a small imperceptible perturbation to an original image to create a new image with the goal of crafting an image to maximize the model's prediction error or loss.

# __Fast Gradient Sign Method (FGSM)__:
#     (based on this intro from tensor flow: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
#     FGSM is a fast method for generating adversarial examples. It works by exploiting how the model's loss function behaves to maximize the model's loss by making only a single change to the input image. (each pixel)
#     The process involves the following steps:
#         
# * Calculating the gradient: FGSM first calculates the gradient of the model's loss function with respect to the pixels of the input image to find the direction in which to change the pixel values to achieve the maximum increase in the loss.
#         
# * Finding the sign: Instead of using the gradient values, FGSM only uses the sign of each element in the gradient. meaning for each pixel it determines if the value should be increased (+1) or decreased (-1) to maximize the loss.
#         
# * Apply the perturbation: The sign vector is then multiplied by a small scalar epsilon, that controls the overall magnitude of the perturbation. 
# 
# The formula is:
# 
# ![image.png](attachment:image.png)
# 
# 

# __Projected Gradient Descent (PGD)__:
#     According to this article on medium (https://medium.com/@zachariaharungeorge/unveiling-the-power-of-projected-gradient-descent-in-adversarial-attacks-2f92509dde3c)
#     PGD is an iterative extension of the basic adversarial idea. Instead of taking one large step, PGD takes multiple smaller steps that make it more likely to find a successful perturbation.
# 
# The process consists of the following steps:
# 
# * Initialization: It starts by taking a small and random step away from the original image but still within the allowed area.
# 
# * Iterative Steps: For a number of steps it calculates the gradient just like in FGSM, and the gradient of the loss is calculated with respect to the current pixels of the perturbed image. Then it takes a small step in the direction of the gradient's sign. the size of this step is determined by a step size parameter, denoted as alpha. (for example in the paper, they used a step size of alpha = 1/255 for their PGD attacks.) After each step, the new perturbed image may have moved outside the allowed area around the original image. The projection step forces the image back into this boundary by clipping the perturbation to ensure its magnitude does not exceed the allowed value (epsilon).
#         
# Because PGD is an iterative process that explores the area around the image more thoroughly, it is considered a much stronger attack than FGSM and is a standard benchmark for evaluating a model's adversarial robustness.

# ### 2-1-2. CLIP

# #### 2-1-2-1. CLIP Architecture

# The idea behind CLIP (Contrastive Language-Image Pre-training) is to learn a multi-modal embedding space where images and text that are similar (in context) are located close to one another. To achieve this, the model is composed of two main encoders, an image encoder and a text encoder.
# 
# * __Image Encoder__: This component takes an image as input and outputs a feature vector. In the paper that introduced CLIP two different architectures were considered for this encoder:
# 
#     A modified version of the ResNet-50 architecture. Where the key modifications included using ResNet-D improvements, antialiased rect-2 blur pooling, and replacing the global average pooling layer with an attention pooling mechanism.
# 
#     And a Vision Transformer (ViT) as an alternative architecture that processed an image as a sequence of flattened patches.
# 
# * __Text Encoder__: This component takes a text as input and generates a corresponding feature vector. The architecture used is a 63M-parameter transformer with 12 layers and 8 attention heads.
# 
# Both encoders are trained jointly from scratch to project their outputs into the same embedding space. A summary of the model architecture can bee seen in the image below:
# 
# ![image.png](attachment:image.png)

# #### 2-1-2-2. Contrastive Training

# Instead of using a traditional objective, like trying to predict the exact text caption for an image, CLIP uses a contrastive objective. The goal isn't to generate content, but rather to match existing images and texts from a large batch.
# 
# The process, has the following steps (as shown in the image of last section):
# 
# * __Create a Batch__
#     
# * __Encode Everything__
# 
# * __Calculate Similarity__: The model computes the cosine similarity between every image embedding and every text embedding. This creates a large N x N matrix of similarity scores .
# 
# * __The Contrastive Goal__: Along the diagonal of this matrix lie the similarity scores for the correct (image, text) pairs and the remaining entries in the matrix are the similarity scores for all the incorrect pairings. The model's training objective is to maximize the similarity of the N correct pairs while minimizing the similarity of the rest of the pairs. 
# 
# In a nutshell the model it learns to associate an image with its correct caption from.

# #### 2-1-2-3. Loss Function

# The model uses Symmetric Cross-Entropy Loss Function
# 
# To accomplish the contrastive training described above, CLIP uses what the paper it was introduced in calls a "symmetric cross entropy loss". This is detailed in the image below.
# 
# ![image.png](attachment:image.png)
# 
# The "symmetric" part means the loss is calculated from two perspectives, and then averaged:
# 
# * __Image-to-Text Loss (loss_i)__: For each image in the batch, this loss treats the task as a classification problem. The goal is to correctly classify which of the text snippets is the true match and standard cross-entropy loss is used to optimize this prediction.
# 
# * __Text-to-Image Loss (loss_t)__: For each text in the batch, the model must correctly classify which of the N images is its true partner. This also uses a cross-entropy loss.
# 
# The final loss for the entire batch is the average of these two individual losses:
# 
# loss = (loss_i + loss_t) / 2 

# ### 2-1-3. Zero-Shot vs Normal Classification

# The primary difference is in how the models are trained and what they can predict.
# 
# __Normal (Supervised) Classification__ is the traditional approach in computer vision where a model is trained on a dataset with a fixed and predetermined set of labels. So to classify a new category, the model must be retrained or fine-tuned with additional labeled examples for that specific category. The output is typically a static softmax classifier that predicts from the limited set of classes it has seen during training.
# 
# __Zero-Shot Classification__ is an approach that allows a model to be much more flexible and general. With this approach a model can classify images into unseen categories that it was not explicitly trained on. In order to achieve this instead of learning from a fixed set of labels, it learns a more general association between visual data and descriptive language. This enables the model to transfer its knowledge to new tasks and datasets without needing additional dataset-specific training.
# 
# Basically, a normally trained classifier can only identify what it has already been taught to see, while a zero-shot classifier can identify new concepts described to it through language.

# CLIP achieves zero-shot classification similarly by using its jointly trained image and text encoders to create a classifier from natural language descriptions at test time.
# 
# The process is illustrated in parts (2) and (3) of figure below:
# 
# ![image.png](attachment:image.png)
# 
# (2) Create dataset classifier from label text
# (3) Use for zero-shot prediction
# 
# Here is a step-by-step breakdown of this process:
# 
# __Create the Classifier from Text__ (Part 2 of Diagram): For any given classification task, the names of all possible target classes are first defined (e.g., "plane", "car", "dog"). Then to provide context, these class names are embedded into prompt templates, such as "A photo of a {label}.". These descriptive prompts are then fed into CLIP's text encoder to generate a set of text embeddings (feature vectors). These embeddings then become the weights of a new, "zero-shot" linear classifier.
# 
# __Encode the image__ (Part 3 of Diagram): The input image that needs to be classified is passed through CLIP's Image Encoder to produce a single image embedding.
# 
# Compare Embeddings for Prediction (Part 3 of Diagram): The model calculates the cosine similarity between the image embedding and all of the text embeddings (the classifier weights) created in step 1.
# 
# __Final Classification__: The class corresponding to the text prompt with the highest cosine similarity is the model's final prediction. Because this classifier is synthesized "on-the-fly" using natural language, CLIP can perform classification on arbitrary new tasks without being retrained.

# ### 2-1-4. Adversarial Attack Types

# In the field of machine learning security, adversarial attacks are categorized based on how much information the attacker has about the target model. The two primary categories are white-box and black-box.
# 
# * __White-Box Attacks__: In this scenario, the attacker has full knowledge of the target model. This includes having the source code, the model's architecture, its parameters and the loss function used for training.Because of this complete access, attackers can use powerful gradient-based methods like the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to precisely calculate the most efficient perturbation to fool the model.
# 
# * __Black-Box Attacks__: In this scenario, the attacker has no access to the model's internal workings. The model is treated as a "black box". The attacker can only provide an input to the model and observe the corresponding output (like the predicted class label and confidence score). Since the attacker cannot directly calculate the model's gradients, they must use other strategies. Common black-box methods include query-based attacks (making many queries to infer the model's decision boundary) and transfer attacks, which are a significant threat in real-world applications.
# 
# We can see a detailed table comparing these two (generated by and AI based on the description above) 
# 
# | Aspect       | White-Box Attacks                                                                                     | Black-Box Attacks                                                                                                             |
# |--------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
# | Knowledge    | Complete access to the model's architecture, parameters, and gradients.                               | No internal knowledge; only input-output access is available.                                                               |
# | Effectiveness| Highly effective. These attacks are optimized for the specific model and represent a "worst-case" scenario for testing robustness. | Less effective than white-box attacks. The success rate is lower and depends heavily on the attack strategy (e.g., the quality of a substitute model in a transfer attack). |
# | Methodology  | Primarily use gradient-based methods like FGSM and PGD to craft perturbations.                        | Rely on query-based or transfer-based methods to estimate or bypass the need for gradients.                                 |
# 
# 

# ### 2-1-5. Transfer Attacks

# Based on this article on medium (https://medium.com/google-developer-experts/cybersecurity-in-ai-transfer-learning-as-an-attack-vector-a6703b017337):
# 
# While white-box attacks are more powerful against a specific model, their real-world application are limited. Transfer attacks, however, represent a more serious and practical threat for the following reasons:
# 
# 1.__Real-World Systems are Black Boxes__: In most real-world scenarios, models deployed by companies are proprietary. Attackers do not have access to the model's architecture, parameters, or gradients. This makes a direct white-box attack, which requires this internal knowledge, impossible. Transfer attacks are a method for attacking these realistic black-box systems.
# 
# 2.__No Internal Access Required__: A transfer attack bypasses the need for internal knowledge. The attacker can simply train their own local model (which is a white-box to them). use powerful white-box methods like PGD to generate an adversarial example that fools their own model and transfer this same adversarial example to the target black-box system.
# 
# 3.__Vulnerabilities are General, Not Specific__: The core reason transfer attacks work is that adversarial examples often exploit fundamental features or flaws in how neural networks learn, rather than targeting a single model's specific weights. Different models trained on similar tasks tend to learn similar feature representations, making them vulnerable to the same adversarial perturbations. According to the paper research has shown that adversarial properties are transferable across different models and tasks.

# ### 2-1-6. LORA

# LoRA, which stands for Low-Rank Adaptation, is a parameter-efficient fine-tuning method designed to make the adaptation of large scale language models to specific downstream tasks more manageable and less computationally expensive.
# 
# The core idea is based on the hypothesis that the change in the weights of a model during adaptation has a low intrinsic rank meaning that the update to the weight matrix doesn't need to be a full-rank matrix and it can be effectively represented by a much smaller number of parameters. Instead of retraining all the parameters of a large pre-trained model (full fine-tuning), LoRA works as by using the following steps:
# 
# 1.  __Freeze Pre-trained Weights__: The majority of the model's original, pre-trained weights are frozen and do not receive any gradient updates.
# 
# 2.  __Inject Low-Rank Matrices__: For a given weight matrix in the model, LoRA injects a pair of smaller, trainable "rank decomposition" matrices, referred to as A and B.
# 
# 3.  __Train Only the Injected Matrices__: During fine-tuning, only these newly added low-rank matrices A and B are trained. The product of these two smaller matrices, BA, represents the change (ΔW) to the original weight matrix (W0). The final output is the sum of the original weights and the learned adaptation: h = W₀x + BAx.
# 
# 4.  __Merge for Deployment__: After training, the learned matrix BA can be merged with the original weight matrix W0 by simple addition (W = W0 + BA) to get the final, adapted weights. This means no extra parameters or calculations are needed during inference.
# 
# This can be seen in the picture below (from LoRA paper)
# 
# ![image.png](attachment:image.png)
# 
# 
# __Three Reasons for Using LoRA__
# 
# The LoRA paper highlights several key advantages of LoRA compared to other methods like full fine-tuning or adapter methods. Here are three of them:
# 
# 1.  __It is extremely parameter-efficient.__ LoRA drastically reduces the number of trainable parameters required for task adaptation. Compared to fully fine-tuning GPT-3 175B, LoRA can reduce the number of trainable parameters by a factor of 10,000. This also reduces the checkpoint size from hundreds of gigabytes to a few megabytes.
# 
# 2.  __It introduces no additional inference latency.__ Unlike adapter methods, which insert new layers that must be processed sequentially and thus add latency, LoRA is designed to be latency-free. Because the learned matrices (BA) can be merged directly into the original weights (W0) after training, the deployed model has the exact same number of parameters and the same architecture as the original pre-trained model.
# 
# 3.  __It makes training more efficient and accessible.__ LoRA can speed up training and lower the hardware barrier to entry. On GPT-3 175B, LoRA reduces the VRAM requirement by a factor of 3 and can result in a 25% training speedup compared to full fine-tuning, as gradients do not need to be computed for the vast majority of frozen parameters. This makes the fine-tuning process faster and accessible on less powerful hardware.

# ### 2-1-7. Two Papers that Expand CLIP

# #### 2-1-7-1. Paper 1: Post-pre-training for Modality Alignment in Vision-Language Foundation Models

# This paper introduces CLIP-Refine, a post-pre-training method designed to improve the performance of existing CLIP models by addressing the modality gap (the discrepancy between how images and text are clustered in the feature space). The authors propose a lightweight training phase that occurs after the initial pre-training but before any fine-tuning. This method uses two novel techniques: __Random Feature Alignment (RaFA)__ and __Hybrid Contrastive-Distillation (HyCD)__. RaFA encourages the image and text features to align to a common distribution, while HyCD uses a mix of ground-truth labels and knowledge from the original CLIP model to learn new information without forgetting what was previously learned. This approach requires minimal training, needing only one epoch on small datasets.
# 
# The CLIP-Refine loss function significantly improves CLIP's zero-shot performance across various tasks. By mitigating the modality gap and enhancing feature uniformity, the model becomes more effective at generalizing to new, unseen data. Experiments show that CLIP-Refine consistently outperforms the standard pre-trained CLIP model in zero-shot classification and retrieval tasks. For instance, on 12 classification datasets, CLIP-Refine increased the average accuracy from 52.74% to 54.69%. The method not only reduces the gap between image and text features but also improves the overall quality of the feature space, leading to more robust and accurate zero-shot capabilities.

# #### 2-1-7-2. Paper 2: Post-pre-training for Modality Alignment in Vision-Language Foundation Models

# This paper introduces __FairerCLIP__, a method designed to mitigate biases in the zero-shot predictions of CLIP models. It addresses biases stemming from both intrinsic dependencies (like correlations between gender and facial features) and spurious correlations (like image backgrounds). The core of FairerCLIP is a debiasing process that operates in a mathematical space known as a Reproducing Kernel Hilbert Space (RKHS). This approach allows it to define an objective function that explicitly aims to reduce the statistical dependence on sensitive attributes (like race or gender) while preserving information about the target attribute and maintaining the alignment between image and text features. A key advantage is its flexibility; it can be trained with or without ground-truth labels and is computationally efficient, leveraging closed-form solutions for optimization.
# 
# The FairerCLIP method effectively reduces bias while maintaining or even improving CLIP's zero-shot classification accuracy. By making the model's predictions more robust and fair, it enhances the reliability of its zero-shot capabilities. For instance, in experiments on datasets with intrinsic dependencies, FairerCLIP drastically reduced the Equal Opportunity Difference (EOD) fairness metric to near-zero while keeping classification accuracy high. Similarly, on datasets with spurious correlations, it consistently improved the worst-group accuracy and reduced the performance gap between different demographic groups. This demonstrates that the debiasing function not only makes predictions fairer but also leads to more robust performance in challenging zero-shot scenarios.

# ## 2-2. Implementation and Comparison of Adversarial Training Methods

# ### 2-2-1. Data Preparation

# Here we will download the data of CIFAR-10 dataset using torchvision, split it into training validation and test sets, transform them for CLIP and display a batch of random sample from the data.

# #### 2-2-1-1. Downloading & Transforming the Data

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms


clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std),
])

full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform)

train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform)

print(f"Full training set size: {len(full_train_dataset)}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")


batch_size = 32 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


# #### 2-2-1-2. Visualising the Transformed and Untransformed data

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def show_random_samples(data_loader, label):
    images, labels = next(iter(data_loader))

    image_grid = torchvision.utils.make_grid(images[:8], nrow=4)

    image_grid = image_grid.numpy().transpose((1, 2, 0))
    image_grid = np.array(clip_std) * image_grid + np.array(clip_mean)
    image_grid = np.clip(image_grid, 0, 1)

    plt.figure(figsize=(24, 12))
    plt.imshow(image_grid)
    plt.title(label)
    plt.axis('off')

    print('Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
    plt.show()

show_random_samples(train_loader, "Random Transformed Samples from CIFAR-10")


# ### 2-2-2. Model Preparation

# Here we will load the pre-trained CLIP model from Hugging Face transformers library, load the pre-trained CIFAR-10 ResNet-20 model from pytorch hub, set both models to eval mode, create the descriptive text prompt for CLIP based on CIFAR-10 class names and use CLIP to process these prompts and generate the text feature vectors needed for zero-shot classification.

# #### 2-2-2-1. Setting up the Device & Loading CLIP

# In[4]:


from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

clip_model.eval()


# #### 2-2-2-2. Loading ResNet-20 model

# In[5]:


target_model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar10_resnet20",
    pretrained=True
).to(device)

target_model.eval()


# #### 2-2-2-3. Preparing the Textual Space for CLIP

# In[6]:


cifar_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

text_prompts = [f"a photo of a {c}" for c in cifar_classes]

with torch.no_grad():
    inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    text_features = clip_model.get_text_features(**inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Shape: {text_features.shape}")


# 
