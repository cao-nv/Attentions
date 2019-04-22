# Attentions module on CIFAR100 for ResNet34 and ResNet50
Cao Nguyen-Van
20184172

# This work is organized as followings:
```
Attentions
|       README.md: This file
|       main.py: File to train the models
|_______utils: Folder for data augmentation, datasets, etc
|       |      data_augmentation.py: Data augmentation functions
|       |      datasets.py: Datasets
|       |      utils.py: Other utils
|_______models: Folder for networks and modules
|       |      attention_resnet.py: ResNet with additional arguments for attention modules
|       |      attention_modules.py: attention modules
|_______configs: Folder for configuration
|       |      config.yaml: configuration for training and testing network
|_______datasets: CIFAR100 dataset
|_______checkpoint: Default folder for logs and saved models
        
```