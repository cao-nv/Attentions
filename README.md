# Attentions module: SE, BAM and CBAM on CIFAR100 for ResNet34 and ResNet50
Cao Nguyen-Van\
`nguyenvancao@kaist.ac.kr`\
20184172

# Requirements:
* Python 2.7 (only 2.7)
* pytorch 1.0.1
* tqdm, argparse, yaml
* ...
# This work is organized as followings:
```
Attentions
|       README.md: This file
|       main.py: File to train the models
|       grad_cam.py File for generating Grad-CAM images 
|_______utils: Folder for dataset and other functions 
|       |      datasets.py: Datasets
|       |      utils.py: Other utils
|_______models: Folder for networks and modules
|       |      attention_resnet.py: ResNet with additional arguments for attention modules
|       |      attention_modules.py: attention modules
|       |      resnet.py: Original ResNet implementation from Torchvision
|       |      models.py: wrapper to create models 
|_______configs: Folder for configuration
|       |      config.yaml: configurations for training and testing network
|_______datasets: CIFAR100 dataset
|_______checkpoint: Default folder for logs and saved models
        
```

# To run train the model:
```
python main.py --arch resnet50 --attention CBAM
```
* `--arch`: `resnet34` or `resnet50`
* `--attention`: 'None', `SE`, `BAM`, `CBAM`, `BAMspatial`, `BAMchannel`, `CBAMspatial`, `CBAMchannel`.

# To run the grad-cam code:
```
python grad_cam.py --arch resnet50 --attention CBAM --output-dir gradCAM_images
```
* `--arch`: `None`, `resnet34` or `resnet50`. If `None`, the original input images will be produced 
* `--attention`: 'None', `SE`, `BAM`, `CBAM`, `BAMspatial`, `BAMchannel`, `CBAMspatial`, `CBAMchannel`.
* `--output-dir`: Output folder to save the images
This script will take the first 8 images from the test set and compute Grad-CAM image with the specified model. In case of `None` in the `--arch` option, the output will be saved in the format `<index>_<class name>.png`. In other cases, the output will be saved in the format `<index>_<archirecture>_<attention>_<probability of the true label>.png`. 
