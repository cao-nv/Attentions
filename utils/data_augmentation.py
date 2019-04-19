from torchvision.transforms import transforms

RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)

Augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)
])

Normalization = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)])


def augment_data(images, augment=True):
    if augment:
        augmented_images = Augmentation(images)
        return augmented_images
    else:
        normalized_images = Normalization(images)
        return normalized_images
