from torchvision import transforms
import torch
from torch.utils.data import Dataset

# Transform (data augmentation) configuration values
IMAGE_SIZE = 227  # For AlexNet
resize_params = (IMAGE_SIZE, IMAGE_SIZE)
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]  # Best values for CIFAR-10
normalize_params = {"mean": mean, "std": std}

random_rotation_params = 20
random_flip_params = 0.1
color_jitter_params = {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1}
random_sharpness_params = {"sharpness_factor": 2, "p": 0.1}
random_erasing_params = {"p": 0.75, "scale": (0.02, 0.1), "value": 1.0}

# Define augmentation configurations for each method
transform_configs = {
    1: {  # Resize
        'resize': True,
        'random_rotation': False,
        'random_flip': False,
        'color_jitter': False,
        'random_sharpness': False,
        'normalize': True,
        'random_erasing': False,
    },
    2: {  # Random Rotation
        'resize': True,
        'random_rotation': random_rotation_params,
        'random_flip': False,
        'color_jitter': False,
        'random_sharpness': False,
        'normalize': True,
        'random_erasing': False,
    },
    3: {  # Random Flip
        'resize': True,
        'random_rotation': False,
        'random_flip': random_flip_params,
        'color_jitter': False,
        'random_sharpness': False,
        'normalize': True,
        'random_erasing': False,
    },
    4: {  # Color Jitter
        'resize': True,
        'random_rotation': False,
        'random_flip': False,
        'color_jitter': color_jitter_params,
        'random_sharpness': False,
        'normalize': True,
        'random_erasing': False,
    },
    5: {  # Random Sharpness
        'resize': True,
        'random_rotation': False,
        'random_flip': False,
        'color_jitter': False,
        'random_sharpness': random_sharpness_params,
        'normalize': True,
        'random_erasing': False,
    },
    6: {  # Random Erasing
        'resize': True,
        'random_rotation': False,
        'random_flip': False,
        'color_jitter': False,
        'random_sharpness': False,
        'normalize': True,
        'random_erasing': random_erasing_params,
    },
    7: {  # Everything
        'resize': True,
        'random_rotation': random_rotation_params,
        'random_flip': random_flip_params,
        'color_jitter': color_jitter_params,
        'random_sharpness': random_sharpness_params,
        'normalize': True,
        'random_erasing': random_erasing_params,
    }
}

def build_transforms(config):
    """
    Dynamically builds a transform pipeline based on the given configuration dictionary.
    """
    transform_list = []
    
    if config.get('resize'):
        transform_list.append(transforms.Resize(resize_params))
    if config.get('random_rotation'):
        transform_list.append(transforms.RandomRotation(config['random_rotation']))
    if config.get('random_flip'):
        transform_list.append(transforms.RandomHorizontalFlip(config['random_flip']))
    if config.get('color_jitter'):
        transform_list.append(transforms.ColorJitter(**config['color_jitter']))
    if config.get('random_sharpness'):
        transform_list.append(transforms.RandomAdjustSharpness(**config['random_sharpness']))
    transform_list.append(transforms.ToTensor())
    if config.get('normalize'):
        transform_list.append(transforms.Normalize(**normalize_params))
    if config.get('random_erasing'):
        transform_list.append(transforms.RandomErasing(**config['random_erasing']))
    
    return transforms.Compose(transform_list)

class CustomCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        """
        A wrapper for applying multiple transforms or a single transform to CIFAR-10 dataset.
        
        Args:
            dataset: Base dataset (CIFAR-10 subset)
            transform: Either a single transform (callable) or a list of transforms (list of callables).
        """
        self.dataset = dataset
        self.transform = transform if isinstance(transform, list) else [transform]

    def __len__(self):
        return len(self.dataset) * len(self.transform)

    def __getitem__(self, index):
        dataset_index = index % len(self.dataset)
        transform_index = index // len(self.dataset)
        img, label = self.dataset[dataset_index]

        if self.transform[transform_index] is not None:
            img = self.transform[transform_index](img)

        return img, label