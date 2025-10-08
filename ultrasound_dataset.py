import os
import json
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
import medmnist
from medmnist import INFO, Evaluator
import requests
from zipfile import ZipFile
import pandas as pd
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from skimage.util import random_noise
import argparse

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)

def build_dataset(args):
    train_transform, test_transform = build_transform(args)
    
    if args.dataset == 'knee':
        data_dir = './data_place'
        print(f"Dataset is available at: {data_dir}")
        full_dataset = datasets.ImageFolder(root=data_dir)  # Load without transform
        # Define the sizes for the splits (total: 4058 images)
        # train_size = 234  # 80%
        # val_size = 30     # 10%
        # test_size = 30    # 10%
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size  # phần còn lại
        nb_classes = 2     # normal vs abnormal
        # nb_classes = 5
        # Verify the total number of images matches the sum of the splits
        assert train_size + val_size + test_size == len(full_dataset), "The sum of the splits must equal the total number of images"
        
    else:
        raise NotImplementedError()

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply the transformations
    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_dataset.indices)
    val_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), val_dataset.indices)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=test_transform), test_dataset.indices)

    print("Number of the class = %d" % nb_classes)

    return train_dataset, test_dataset, nb_classes

class UltrasoundPreprocessor:
    """Custom preprocessing for ultrasound images"""
    
    @staticmethod
    def remove_borders_and_annotations(image):
        """Remove black borders and annotations from ultrasound images"""
        img_array = np.array(image)
        # Convert to grayscale for border detection only
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        coords = cv2.findNonZero(img_gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2*padding, img_gray.shape[1] - x)
            h = min(h + 2*padding, img_gray.shape[0] - y)
            if len(img_array.shape) == 3:
                cropped = img_array[y:y+h, x:x+w, :]
            else:
                cropped = img_array[y:y+h, x:x+w]
            return Image.fromarray(cropped)
        return image

    @staticmethod
    def apply_clahe(image):
        """Apply CLAHE to each channel if RGB, else to grayscale"""
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            channels = [clahe.apply(img_array[:,:,i]) for i in range(3)]
            enhanced = np.stack(channels, axis=2)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_array)
        return Image.fromarray(enhanced)

class UltrasoundAugmentation:
    """Custom augmentation for ultrasound images"""
    
    @staticmethod
    def random_rotation(image, degrees=15):
        """Apply random rotation within specified degrees"""
        angle = np.random.uniform(-degrees, degrees)
        return image.rotate(angle, fillcolor=0)
    
    @staticmethod
    def random_horizontal_flip(image, probability=0.5):
        """Apply random horizontal flip with 50% probability"""
        if np.random.random() < probability:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image
    
    @staticmethod
    def random_translation(image, max_shift=0.1):
        """Apply random translation"""
        width, height = image.size
        max_x_shift = int(width * max_shift)
        max_y_shift = int(height * max_shift)
        
        x_shift = np.random.randint(-max_x_shift, max_x_shift)
        y_shift = np.random.randint(-max_y_shift, max_y_shift)
        
        # Create new image with black background
        new_image = Image.new(image.mode, image.size, 0)
        new_image.paste(image, (x_shift, y_shift))
        return new_image
    
    @staticmethod
    def random_scaling(image, scale_range=(0.9, 1.1)):
        """Apply random scaling (zoom in/out)"""
        scale = np.random.uniform(*scale_range)
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop or pad to original size
        if scale > 1:
            # Crop from center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop((left, top, left + width, top + height))
        else:
            # Pad with black
            new_image = Image.new(image.mode, image.size, 0)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_image.paste(scaled, (left, top))
            return new_image
    
    @staticmethod
    def random_shear(image, max_shear=10):
        """Apply random shear transformation"""
        shear_x = np.random.uniform(-max_shear, max_shear)
        shear_y = np.random.uniform(-max_shear, max_shear)
        
        # Convert to numpy for shear
        img_array = np.array(image)
        rows, cols = img_array.shape[:2]
        
        # Create shear matrix
        M = np.array([[1, shear_x/100, 0], [shear_y/100, 1, 0]], dtype=np.float32)
        
        if len(img_array.shape) == 3:
            sheared = cv2.warpAffine(img_array, M, (cols, rows), borderValue=(0,0,0))
        else:
            sheared = cv2.warpAffine(img_array, M, (cols, rows), borderValue=[0])
        return Image.fromarray(sheared)
    
    @staticmethod
    def random_brightness_contrast(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        """Apply random brightness and contrast adjustments"""
        # Brightness
        brightness_factor = np.random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        # Contrast
        contrast_factor = np.random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image
    
    @staticmethod
    def random_gamma(image, gamma_range=(0.7, 1.3)):
        """Apply random gamma correction"""
        gamma = np.random.uniform(*gamma_range)
        img_array = np.array(image, dtype=np.float32) / 255.0
        corrected = np.power(img_array, gamma) * 255.0
        return Image.fromarray(corrected.astype(np.uint8))
    
    @staticmethod
    def add_speckle_noise(image, intensity=0.1):
        """Add speckle noise to simulate ultrasound characteristics"""
        img_array = np.array(image, dtype=np.float32) / 255.0
        noisy = random_noise(img_array, mode='speckle', var=intensity)
        noisy = np.clip(noisy, 0, 1) * 255.0
        return Image.fromarray(noisy.astype(np.uint8))
    
    @staticmethod
    def random_erasing(image, probability=0.3, area_ratio=(0.02, 0.2)):
        """Apply random erasing/cutout"""
        if np.random.random() > probability:
            return image
        
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Random area
        area = height * width
        target_area = np.random.uniform(*area_ratio) * area
        
        # Random aspect ratio
        aspect_ratio = np.random.uniform(0.3, 3.33)
        
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h < height and w < width:
            top = np.random.randint(0, height - h)
            left = np.random.randint(0, width - w)
            
            # Fill with black
            img_array[top:top + h, left:left + w] = 0
        
        return Image.fromarray(img_array)

class UltrasoundTransform:
    """Combined transform for ultrasound images"""
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.preprocessor = UltrasoundPreprocessor()
        self.augmenter = UltrasoundAugmentation()
    
    def __call__(self, image):
        image = self.preprocessor.remove_borders_and_annotations(image)
        if np.random.random() > 0.5:
            image = self.preprocessor.apply_clahe(image)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        if self.is_training:
            if np.random.random() > 0.5:
                image = self.augmenter.random_rotation(image, degrees=15)
            if np.random.random() > 0.5:
                image = self.augmenter.random_horizontal_flip(image)
            if np.random.random() > 0.5:
                image = self.augmenter.random_translation(image, max_shift=0.1)
            if np.random.random() > 0.5:
                image = self.augmenter.random_scaling(image, scale_range=(0.9, 1.1))
            if np.random.random() > 0.5:
                image = self.augmenter.random_shear(image, max_shear=10)
            if np.random.random() > 0.5:
                image = self.augmenter.random_brightness_contrast(image)
            if np.random.random() > 0.5:
                image = self.augmenter.random_gamma(image)
            if np.random.random() > 0.5:
                image = self.augmenter.add_speckle_noise(image, intensity=0.1)
            if np.random.random() > 0.5:
                image = self.augmenter.random_erasing(image, probability=0.3)
        image = transforms.ToTensor()(image)
        image = (image - 0.5) * 2  # Scale from [0,1] to [-1,1]
        return image

def build_transform(args):
    """Build transforms for ultrasound images"""
    
    # Training transform with all augmentations
    train_transform = UltrasoundTransform(is_training=True)
    
    # Test transform (preprocessing only)
    test_transform = UltrasoundTransform(is_training=False)
    
    return train_transform, test_transform 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="knee")
    args = args.parse_args()
    train_dataset, test_dataset, nb_classes = build_dataset(args)
    print(train_dataset[0])
    print(test_dataset[0])
    print(nb_classes)