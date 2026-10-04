#!/usr/bin/env python
# coding: utf-8

# In the name of God

# # Q1: Analysis of Neural Network Performance Under Adversarial Attacks

# In this notebook we will be training a few CNN and ViT networks and test their performance on noisy data and under adversarial attacks, defend against such attacks by using adversarial training and answer a few questions.

# ## 1-1. Training ResNet on noisy images

# In this section we will be using a ResNet model (ResNet34) and train it on the CIFAR-100 dataset with and without noise to investigate the effects of noise on model performance.

# ### 1-1-1. Loading the dataset

# As the first step we will use pytorch to load the dataset, normalise it (for the use in the network) and define the appropriate data loaders

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms

transform_clean = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

batch_size = 128

train_set_clean = torchvision.datasets.CIFAR100(
    root='./data', 
    train=True,
    download=True, 
    transform=transform_clean
)
train_loader_clean = torch.utils.data.DataLoader(
    train_set_clean, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)

test_set = torchvision.datasets.CIFAR100(
    root='./data', 
    train=False,
    download=True, 
    transform=transform_clean
)

test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=2
)

print(len(train_set_clean))
print(len(test_set))


# ### 1-1-2. Adding noise

# In order to add the noise, we will define a class called AddGaussianNoise that can be used in the transformation part (when we are loading the data) and add the noise to the data (before we normalise it)
# 
# In order to verify the process we will visualise a random image and its noisy counterpart.

# In[2]:


import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

noise_mean = 0
noise_variance = 0.005
noise_std_dev = np.sqrt(noise_variance)

transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=noise_mean, std=noise_std_dev),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_set_noisy = torchvision.datasets.CIFAR100(
    root='./data', 
    train=True,
    download=True, 
    transform=transform_noisy
)

train_loader_noisy = torch.utils.data.DataLoader(
    train_set_noisy, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)


# In[3]:


import matplotlib.pyplot as plt

def display_image(image, axes):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    image = image.numpy().transpose((1, 2, 0)) 
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    axes.imshow(image)
    axes.axis('off')

clean_images, _ = next(iter(train_loader_clean))
noisy_images, _ = next(iter(train_loader_noisy))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].set_title('Original Image')
display_image(clean_images[0], axes=axes[0])

axes[1].set_title('Image with Gaussian Noise')
display_image(noisy_images[0], axes=axes[1])

plt.show()


# ### 1-1-3. Training the ResNet model

# In this section we will use the ResNet34 in the pytorch library (from torchvision.models), define a function to loop and train the model for 20 epochs and save the results to be plotted in the next section. We will do this process once with the normal data and once with the noisy data and compare the results.
# 
# We will also try different hyper parameter tuning methods, such as L2 regularization, adding a dropout layer, learning rate scheduler and data augmentation

# In[4]:


import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from time import time
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_validate(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion, 
        epochs,
        scheduler=None,
        keep_best=False
    ):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0
    best_model_state = None

    start_time = time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0 
        total_train = 0 

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_acc = 100 * correct_val / total_val
        epoch_val_loss = val_loss / len(test_loader)
        history['val_acc'].append(epoch_val_acc)
        history['val_loss'].append(epoch_val_loss)

        if keep_best and epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch + 1}: New best model saved with Val Acc: {best_val_acc:.2f}%")
            
        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.2f}% | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {epoch_val_acc:.2f}%')

    end_time = time()
    print(f'Finished Training. Total time: {(end_time - start_time)/60:.2f} minutes')
    
    if keep_best:
        return history, best_model_state
    
    return history


# In[5]:


epochs = 20
learning_rate = 0.001
num_classes = 100

model_clean_plain = models.resnet34(weights=None)
model_clean_plain.fc = nn.Linear(model_clean_plain.fc.in_features, num_classes)
model_clean_plain.to(device)

model_noisy_plain = models.resnet34(weights=None)
model_noisy_plain.fc = nn.Linear(model_noisy_plain.fc.in_features, num_classes)
model_noisy_plain.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_clean_plain = optim.Adam(model_clean_plain.parameters(), lr=learning_rate)
optimizer_noisy_plain = optim.Adam(model_noisy_plain.parameters(), lr=learning_rate)


print("Training the model on clean data")
history_clean_plain = train_and_validate(
    model_clean_plain, 
    train_loader_clean,
    test_loader,
    optimizer_clean_plain,
    criterion,
    epochs
)

print()
print("Training the model on noisy data")
history_noisy_plain = train_and_validate(
    model_noisy_plain,
    train_loader_noisy,
    test_loader,
    optimizer_noisy_plain,
    criterion,
    epochs
)


# As we can see, the models have clearly overfitted.
# 
# In order to see if we can improve the models, next we will add L2 regularization to our models.

# In[6]:


epochs = 20
learning_rate = 0.001
num_classes = 100

model_clean_l2 = models.resnet34(weights=None)
model_clean_l2.fc = nn.Linear(model_clean_l2.fc.in_features, num_classes)
model_clean_l2.to(device)

model_noisy_l2 = models.resnet34(weights=None)
model_noisy_l2.fc = nn.Linear(model_noisy_l2.fc.in_features, num_classes)
model_noisy_l2.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_clean_l2 = optim.Adam(
    model_clean_l2.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
optimizer_noisy_l2 = optim.Adam(
    model_noisy_l2.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)


print("Training the model on clean data")
history_clean_l2 = train_and_validate(
    model_clean_l2, 
    train_loader_clean,
    test_loader,
    optimizer_clean_l2,
    criterion,
    epochs
)

print()
print("Training the model on noisy data")
history_noisy_l2 = train_and_validate(
    model_noisy_l2,
    train_loader_noisy,
    test_loader,
    optimizer_noisy_l2,
    criterion,
    epochs
)


# Even after adding L2 regularization still both of the models overfitted, with the only difference being that their final accuracy on the training data was reduced, next we will try adding a dropout layer before the final classification layer

# In[7]:


epochs = 20
learning_rate = 0.001
num_classes = 100
dropout_rate = 0.5

model_clean_l2_dropout = models.resnet34(weights=None)
model_clean_l2_dropout.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model_clean_l2_dropout.fc.in_features, 100)
)
model_clean_l2_dropout.to(device)

model_noisy_l2_dropout = models.resnet34(weights=None)
model_noisy_l2_dropout.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model_noisy_l2_dropout.fc.in_features, 100)
)
model_noisy_l2_dropout.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_clean_l2_dropout = optim.Adam(
    model_clean_l2_dropout.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
optimizer_noisy_l2_dropout = optim.Adam(
    model_noisy_l2_dropout.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)


print("Training the model on clean data")
history_clean_l2_dropout = train_and_validate(
    model_clean_l2_dropout, 
    train_loader_clean,
    test_loader,
    optimizer_clean_l2_dropout,
    criterion,
    epochs
)

print()
print("Training the model on noisy data")
history_noisy_l2_dropout = train_and_validate(
    model_noisy_l2_dropout,
    train_loader_noisy,
    test_loader,
    optimizer_noisy_l2_dropout,
    criterion,
    epochs
)


# Dropout didn't have much of an effect either so next we will try using schedulers

# In[8]:


from torch.optim.lr_scheduler import StepLR

epochs = 20
learning_rate = 0.001
num_classes = 100
dropout_rate = 0.5

model_clean_l2_dropout_scheduler = models.resnet34(weights=None)
model_clean_l2_dropout_scheduler.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model_clean_l2_dropout_scheduler.fc.in_features, 100)
)
model_clean_l2_dropout_scheduler.to(device)

model_noisy_l2_dropout_scheduler = models.resnet34(weights=None)
model_noisy_l2_dropout_scheduler.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(model_noisy_l2_dropout_scheduler.fc.in_features, 100)
)
model_noisy_l2_dropout_scheduler.to(device)

criterion = nn.CrossEntropyLoss()

scheduler_clean = StepLR(optimizer_clean_plain, step_size=7, gamma=0.1)
scheduler_noisy = StepLR(optimizer_noisy_plain, step_size=7, gamma=0.1)

optimizer_clean_l2_dropout_scheduler = optim.Adam(
    model_clean_l2_dropout_scheduler.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
optimizer_noisy_l2_dropout_scheduler = optim.Adam(
    model_noisy_l2_dropout_scheduler.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)


print("Training the model on clean data")
history_clean_l2_dropout_scheduler = train_and_validate(
    model_clean_l2_dropout_scheduler, 
    train_loader_clean,
    test_loader,
    optimizer_clean_l2_dropout_scheduler,
    criterion,
    epochs,
    scheduler_clean
)

print()
print("Training the model on noisy data")
history_noisy_l2_dropout_scheduler = train_and_validate(
    model_noisy_l2_dropout_scheduler,
    train_loader_noisy,
    test_loader,
    optimizer_noisy_l2_dropout_scheduler,
    criterion,
    epochs,
    scheduler_noisy
)


# None of the above attempts seems to have improved the model, we could try using data augmentation but since the goal of training these models were to study the effects of noise on model performance, we will suffice to this much.

# ### 1-1-4. Plotting accuracy and loss

# After training the model in order to better visualise their performance, we will plot the accuracy and loss for both the training and test data for all the models we have trained thus far.
# 
# We will present four plots, for accuracy and loss on both the normal data and noisy data and plot all the models in all four plots.

# In[9]:


def plot_metrics(ax, history_dictionary, title):
    epochs_range = range(1, epochs + 1)
    
    colors = {
        'Plain': 'blue',
        'L2': 'green',
        'L2+Dropout': 'red',
        'L2+Dropout+Scheduler': 'purple'
    }
    
    for label, history in history_dictionary.items():
        color = colors[label]
        ax.plot(
            epochs_range, 
            history[0], 
            color=color, 
            linestyle='-', 
            label=f'{label} - Train'
        )
        ax.plot(
            epochs_range, 
            history[1], 
            color=color, 
            linestyle='--', 
            label=f'{label} - Validation'
        )
        
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(title.split(' ')[0])
    ax.legend(loc='best', fontsize='small')
    ax.grid(True)

histories_clean_acc = {
    'Plain': (history_clean_plain['train_acc'], history_clean_plain['val_acc']),
    'L2': (history_clean_l2['train_acc'], history_clean_l2['val_acc']),
    'L2+Dropout': (history_clean_l2_dropout['train_acc'], history_clean_l2_dropout['val_acc']),
    'L2+Dropout+Scheduler': (history_clean_l2_dropout_scheduler['train_acc'], history_clean_l2_dropout_scheduler['val_acc'])
}

histories_clean_loss = {
    'Plain': (history_clean_plain['train_loss'], history_clean_plain['val_loss']),
    'L2': (history_clean_l2['train_loss'], history_clean_l2['val_loss']),
    'L2+Dropout': (history_clean_l2_dropout['train_loss'], history_clean_l2_dropout['val_loss']),
    'L2+Dropout+Scheduler': (history_clean_l2_dropout_scheduler['train_loss'], history_clean_l2_dropout_scheduler['val_loss'])
}

histories_noisy_acc = {
    'Plain': (history_noisy_plain['train_acc'], history_noisy_plain['val_acc']),
    'L2': (history_noisy_l2['train_acc'], history_noisy_l2['val_acc']),
    'L2+Dropout': (history_noisy_l2_dropout['train_acc'], history_noisy_l2_dropout['val_acc']),
    'L2+Dropout+Scheduler': (history_noisy_l2_dropout_scheduler['train_acc'], history_noisy_l2_dropout_scheduler['val_acc'])
}

histories_noisy_loss = {
    'Plain': (history_noisy_plain['train_loss'], history_noisy_plain['val_loss']),
    'L2': (history_noisy_l2['train_loss'], history_noisy_l2['val_loss']),
    'L2+Dropout': (history_noisy_l2_dropout['train_loss'], history_noisy_l2_dropout['val_loss']),
    'L2+Dropout+Scheduler': (history_noisy_l2_dropout_scheduler['train_loss'], history_noisy_l2_dropout_scheduler['val_loss'])
}


fig, axs = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Model Performance Comparison with Different Regularization Techniques', fontsize=20)

plot_metrics(axs[0, 0], histories_clean_acc, 'Accuracy on Clean Data')
plot_metrics(axs[0, 1], histories_clean_loss, 'Loss on Clean Data')

plot_metrics(axs[1, 0], histories_noisy_acc, 'Accuracy on Noisy Data')
plot_metrics(axs[1, 1], histories_noisy_loss, 'Loss on Noisy Data')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# As we can see all he models had quite similar performance on the validation data (which is the important one) so we can't just judge them based on these plots and we are going to use tables to draw the final conclusion in the next section. From these plots one thing that stands out is the fact that all the models have overfitted at around 7 epochs (the loss on the validation data began increasing and the accuracy stayed the same) 

# ### 1-1-5. Comparing the model performances

# To be able to better analyse the performance of the models we are going to get the best results of all the models and put them in a table.

# In[ ]:


import pandas as pd

data_summary = []

histories = {
    ('Clean', '1. Plain'): history_clean_plain,
    ('Clean', '2. L2'): history_clean_l2,
    ('Clean', '3. L2 + Dropout'): history_clean_l2_dropout,
    ('Clean', '4. L2 + Dropout + Scheduler'): history_clean_l2_dropout_scheduler,
    ('Noisy', '1. Plain'): history_noisy_plain,
    ('Noisy', '2. L2'): history_noisy_l2,
    ('Noisy', '3. L2 + Dropout'): history_noisy_l2_dropout,
    ('Noisy', '4. L2 + Dropout + Scheduler'): history_noisy_l2_dropout_scheduler
}

for (data_type, model_name), history in histories.items():
    data_summary.append({
        'Training Data': data_type,
        'Strategy': model_name,
        'Best Val Accuracy (%)': max(history['val_acc']),
        'Best Val Loss': min(history['val_loss'])
    })

data_summary_df = pd.DataFrame(data_summary)

styled_df = data_summary_df.style.format({
    'Best Val Accuracy (%)': '{:.2f}',
    'Best Val Loss': '{:.4f}'
}).hide(axis='index').set_properties(**{'text-align': 'left'})

display(styled_df)


# From this table we can see that the best model on clean data was the one that only used all methods and the best model on the noisy data was the one that used L2, but since the performance on the noisy data between the models were very close between the L2 + Dropout + Scheduler model and L2 model we are going to choose the model that used all the methods for the future parts (we are going to train it once again and keep the best during the training process)
# 
# If we compare the results on clean (45.51%) and noisy (43.89%), we can see despite the noise disturbing the data, the model managed to filter it and the performance drop was less than 2%, this could be due to the fact that ResNet is a rather large model and has a good generalization capability and it can easily filter out noise.
# 
# As for the low performance of the model, we can propose two theories: 1. The model did not have enough input data or 2. The model was not complex enough to be able to separate the data. Since ResNet is a rather big and powerful model the first theory is more likely and in order to improve the performance we could use data augmentation but since it is not the goal of this project we will leave it at that.

# In[11]:


epochs = 20
learning_rate = 0.001
num_classes = 100
dropout_rate = 0.5

resNet_clean = models.resnet34(weights=None)
resNet_clean.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(resNet_clean.fc.in_features, 100)
)
resNet_clean.to(device)

resNet_noisy = models.resnet34(weights=None)
resNet_noisy.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(resNet_noisy.fc.in_features, 100)
)
resNet_noisy.to(device)

criterion = nn.CrossEntropyLoss()


optimizer_clean = optim.Adam(
    resNet_clean.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)
optimizer_noisy = optim.Adam(
    resNet_noisy.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler_noisy = StepLR(optimizer_noisy, step_size=7, gamma=0.1)
scheduler_clean = StepLR(optimizer_clean, step_size=7, gamma=0.1)

print("Training the model on clean data")
history_clean_l2_dropout_scheduler, best_resNet_clean = train_and_validate(
    resNet_clean, 
    train_loader_clean,
    test_loader,
    optimizer_clean,
    criterion,
    epochs,
    scheduler_clean,
    keep_best=True
)
resNet_clean.load_state_dict(best_resNet_clean)

print()
print("Training the model on noisy data")
history_noisy_l2_dropout_scheduler, best_resNet_noisy = train_and_validate(
    resNet_noisy,
    train_loader_noisy,
    test_loader,
    optimizer_noisy,
    criterion,
    epochs,
    scheduler_noisy,
    keep_best=True
)
resNet_noisy.load_state_dict(best_resNet_noisy)


# Removing the old models to free up the vram

# In[12]:


del model_clean_plain
del model_noisy_plain
del model_clean_l2
del model_noisy_l2
del model_clean_l2_dropout
del model_noisy_l2_dropout
del model_clean_l2_dropout_scheduler
del model_noisy_l2_dropout_scheduler

del optimizer_clean_plain
del optimizer_noisy_plain
del optimizer_clean_l2
del optimizer_noisy_l2
del optimizer_clean_l2_dropout
del optimizer_noisy_l2_dropout
del optimizer_clean_l2_dropout_scheduler
del optimizer_noisy_l2_dropout_scheduler

torch.cuda.empty_cache()


# ## 1-2. Transfer learning on Flowers-102 dataset

# ### 1-2-0. Loading and preprocessing the dataset

# We will be using the flowers-102 dataset which is a collection of images of common flowers in UK and contains 102 different classes with over 8,000 images.
# 
# In this section we will download it and prepare the appropriate data loaders and transform the images to work with the ViT, we will also use random Horizontal flips and random rotations to augment the data to improve the performance of the model!

# In[13]:


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_validation_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_flowers = torchvision.datasets.Flowers102(
    root='./data', 
    split='train',
    download=True, 
    transform=transform_train
)

validation_flowers = torchvision.datasets.Flowers102(
    root='./data', 
    split='val',
    download=True, 
    transform=transform_validation_test
)

test_flowers = torchvision.datasets.Flowers102(
    root='./data', 
    split='test',
    download=True, 
    transform=transform_validation_test
)

batch_size = 16
train_loader_flowers = torch.utils.data.DataLoader(
    train_flowers, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)

val_loader_flowers = torch.utils.data.DataLoader(
    validation_flowers, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=2
)

test_loader_flowers = torch.utils.data.DataLoader(
    test_flowers, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=2
)


print(len(train_flowers))
print(len(validation_flowers))
print(len(test_flowers))


# In[14]:


def show_flower_images(image_tensor):
    image_grid = torchvision.utils.make_grid(image_tensor)
    
    np_image = image_grid.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = std * np_image + mean
    np_image = np.clip(np_image, 0, 1)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

images, labels = next(iter(train_loader_flowers))

show_flower_images(images)


# ### 1-2-1. Loading the pre-trained model

# We will load the pre-trained vit_base_patch16_224 model from torchvision and change its output layer to accommodate the 102 classes of our dataset. Since we are going to use transfer learning we are also going to freeze all the parameters in the backbone of the model and only allow the classification head to be trained.

# In[15]:


weights = models.ViT_B_16_Weights.IMAGENET1K_V1
vit_model_pretrained = models.vit_b_16(weights=weights)

for param in vit_model_pretrained.parameters():
    param.requires_grad = False

num_classes = 102
in_features = vit_model_pretrained.heads.head.in_features

vit_model_pretrained.heads = nn.Linear(in_features=in_features, out_features=num_classes)

vit_model_pretrained.to(device)

print(vit_model_pretrained.heads)

total_parameters = sum(parameter.numel() for parameter in vit_model_pretrained.parameters())
trainable_parameters = sum(parameter.numel() for parameter in vit_model_pretrained.parameters() if parameter.requires_grad)

print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# ### 1-2-2. Fine tuning on Flowers-102

# Now we will fine tune the model for 5 epochs on the flowers-102 dataset and keep the best model. 

# In[16]:


epochs_vit_pretrained = 5
learning_rate_ViT_pretrained = 0.001

criterion_vit = nn.CrossEntropyLoss()

parameters_to_update = filter(lambda parameter: parameter.requires_grad, vit_model_pretrained.parameters())
optimizer_vit = optim.Adam(
    parameters_to_update, 
    lr=learning_rate_ViT_pretrained,
    weight_decay=1e-4
)

print("Fine-tuning the ViT model on Flowers-102")

history_vit_pretrained, best_vit_model_pretrained = train_and_validate(
    model=vit_model_pretrained,
    train_loader=train_loader_flowers,
    test_loader=val_loader_flowers,
    optimizer=optimizer_vit,
    criterion=criterion_vit,
    epochs=epochs_vit_pretrained,
    keep_best=True
)

vit_model_pretrained.load_state_dict(best_vit_model_pretrained)


# In order to get the best results after fine tuning the parameters we landed on a learning rate of 0.001 and a regularization factor (weight_decay) of 0.0001
# 
# The pre-trained model managed to get a respectable 80% accuracy on the validation data. Looking at the trend we can see the signs of overfitting so I assume even with more epochs the performance would not improve much. 

# ### 1-2-2. Loading & training the ViT from scratch

# Now we will load the same model but this time without the weights and train the model from scratch on the dataset, originally the model was trained for about 10 epochs but since there was still some room for improvement we let the model to be trained for 30 epochs.

# In[17]:


epochs_vit_scratch = 30
learning_rate_vit_scratch = 0.00001

vit_model_scratch = models.vit_b_16(weights=None)

num_classes = 102
in_features_scratch = vit_model_scratch.heads.head.in_features
vit_model_scratch.heads = nn.Linear(in_features=in_features_scratch, out_features=num_classes)
vit_model_scratch.to(device)

optimizer_vit_scratch = optim.Adam(
    vit_model_scratch.parameters(), 
    lr=learning_rate_vit_scratch,
    weight_decay=1e-4
)

criterion_vit_scratch = nn.CrossEntropyLoss()

print("Training the ViT model from SCRATCH on Flowers-102")

history_vit_scratch, best_vit_model_scratch = train_and_validate(
    model=vit_model_scratch,
    train_loader=train_loader_flowers,
    test_loader=val_loader_flowers,
    optimizer=optimizer_vit_scratch,
    criterion=criterion_vit_scratch,
    epochs=epochs_vit_scratch,
    keep_best=True
)

vit_model_scratch.load_state_dict(best_vit_model_scratch)


# After extensive hyper-parameter tuning the best model was obtained by using L2 regularization with a parameter of 1e-4 and a learning rate of 1e-5.
# 
# After training for 30 epochs and keeping the best model it managed to get an accuracy of 35% on validation data but we can see clear signs of overfitting, which is to be expected given the massive size of the ViT model and the small size of our dataset.

# In[18]:


fig, axs = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('ViT Performance Comparison: Pre-trained vs. From Scratch', fontsize=16)

epochs_pretrained = len(history_vit_pretrained['val_acc'])
epochs_scratch = len(history_vit_scratch['val_acc'])
range_pretrained = range(1, epochs_pretrained + 1)
range_scratch = range(1, epochs_scratch + 1)

ax = axs[0]
ax.plot(range_pretrained, history_vit_pretrained['train_acc'], color='blue', linestyle='-', label='Pre-trained - Train')
ax.plot(range_pretrained, history_vit_pretrained['val_acc'], color='blue', linestyle='--', label='Pre-trained - Val')
ax.plot(range_scratch, history_vit_scratch['train_acc'], color='orangered', linestyle='-', label='From Scratch - Train')
ax.plot(range_scratch, history_vit_scratch['val_acc'], color='orangered', linestyle='--', label='From Scratch - Val')

ax.set_title('Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy (%)')
ax.legend()
ax.grid(True)


ax = axs[1]
ax.plot(range_pretrained, history_vit_pretrained['train_loss'], color='blue', linestyle='-', label='Pre-trained - Train')
ax.plot(range_pretrained, history_vit_pretrained['val_loss'], color='blue', linestyle='--', label='Pre-trained - Val')
ax.plot(range_scratch, history_vit_scratch['train_loss'], color='orangered', linestyle='-', label='From Scratch - Train')
ax.plot(range_scratch, history_vit_scratch['val_loss'], color='orangered', linestyle='--', label='From Scratch - Val')

ax.set_title('Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

plt.show()

print(max(history_vit_pretrained['val_acc']))
print(max(history_vit_scratch['val_acc']))


# We can see that both models have been trained an some signs of overfitting are visible (especially in the from scratch version).
# 
# At the end the pre-trained model got an accuracy of 80% and the scratch model got an accuracy of 36%. This massive gap in performance shows the power of the pre-training, especially in ViTs.
# 
# The pre-trained model has been trained on the ImageNet dataset which is vast library of over a million images across 1000 classes, training the model on this dataset taught it how to extract rich features from the images so when we wanted to apply this model to the our flowers dataset, we only needed to train the classifier portion (and teach the model how to use its rich features to distinguish the flowers).
# 
# In contrast the scratch model only had the limited flowers dataset to train on, since unlike CNN model, vision transformer models have no inherit knowledge of the image, they need to learn the concepts of edges, shapes etc from scratch and this, along with their massive number of parameters causes them to require a lot of data, which in our case was not present. So they mostlikely overfit and have very low performance as we saw.

# 
