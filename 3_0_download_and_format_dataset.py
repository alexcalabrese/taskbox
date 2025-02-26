from datasets import load_dataset
from pathlib import Path
import shutil
import os
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
import logging
from typing import List, Dict, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device resolution and scale mapping
DEVICE_INFO = {
    "default": {
        "scale": 1,
        "resolution": (1280, 720),  # Default 720p
        "aspect_ratio": 16/9
    },
    "iPad-Mini": {
        "scale": 2, 
        "resolution": (1488, 2266),  # Physical resolution
        "aspect_ratio": 2/3
    },
    "iPad-Pro": {
        "scale": 2,
        "resolution": (2048, 2732),  # Physical resolution
        "aspect_ratio": 3/4
    },
    "iPhone-13": {
        "scale": 3,
        "resolution": (1170, 2532),  # Physical resolution
        "aspect_ratio": 9/21
    },
    "iPhone-SE": {
        "scale": 3,
        "resolution": (750, 1334),  # Physical resolution
        "aspect_ratio": 9/16
    }
}

def determine_device(device_key: str) -> Tuple[str, float]:
    """Determine the device and scale factor from the device key.
    
    Parameters
    ----------
    device_key : str
        Device identifier from the dataset
        
    Returns
    -------
    Tuple[str, float]
        Device name and scale factor
    """
    logger.debug(f"Determining device info for key: {device_key}")
    
    # Default values
    device = "default"
    scale = 1.0
    
    # Try to match the device key with known devices
    for known_device in DEVICE_INFO.keys():
        if known_device.lower() in device_key.lower():
            device = known_device
            scale = DEVICE_INFO[device]["scale"]
            logger.debug(f"Matched device: {device} with scale: {scale}")
            break
    
    if device == "default":
        logger.warning(f"No specific device match found for {device_key}, using default settings")
            
    return device, scale

def convert_box_to_yolo_format(box: List[float], image_size: Tuple[int, int], device_key: str) -> List[float]:
    """Convert box coordinates to YOLO format with device-specific scaling.
    
    Parameters
    ----------
    box : List[float]
        Box coordinates [x1, y1, x2, y2]
    image_size : Tuple[int, int]
        Image dimensions (width, height)
    device_key : str
        Device identifier from the dataset
        
    Returns
    -------
    List[float]
        Normalized coordinates [x_center, y_center, width, height]
    """
    logger.debug(f"Converting box {box} with image size {image_size} for device {device_key}")
    
    # Get device info and scale factor
    device, scale = determine_device(device_key)
    
    # Apply device-specific scaling to coordinates
    x1, y1, x2, y2 = [coord * scale for coord in box]
    width, height = image_size
    
    # Log original and scaled coordinates
    logger.debug(f"Original coordinates: {box}")
    logger.debug(f"Scaled coordinates: [{x1}, {y1}, {x2}, {y2}]")
    
    # Ensure coordinates are within bounds
    x1 = min(max(0, x1), width)
    y1 = min(max(0, y1), height)
    x2 = min(max(0, x2), width)
    y2 = min(max(0, y2), height)
    
    # Calculate center points and dimensions
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Normalize
    x_center /= width
    y_center /= height
    box_width /= width
    box_height /= height
    
    yolo_format = [x_center, y_center, box_width, box_height]
    logger.debug(f"YOLO format coordinates: {yolo_format}")
    
    return yolo_format

def create_compound_label(label_list: List[str]) -> str:
    """Create a compound label from a list of labels.
    
    Parameters
    ----------
    label_list : List[str]
        List of labels
        
    Returns
    -------
    str
        Concatenated label with underscore separator, using up to 3 labels
    """
    logger.debug(f"Creating compound label from: {label_list}")
    result = '_'.join(label_list[:3]) if label_list else 'Unknown'
    logger.debug(f"Generated compound label: {result}")
    return result

def is_valid_box(box: tuple, image_size: tuple, min_size: int = 10) -> bool:
    """Check if a bounding box is valid.
    
    Parameters
    ----------
    box : tuple
        Box coordinates (x1, y1, x2, y2)
    image_size : tuple
        Image dimensions (width, height) 
    min_size : int
        Minimum pixel distance between points
        
    Returns
    -------
    bool
        True if box is valid, False otherwise
    """
    logger.debug(f"Validating box {box} with image size {image_size} and min_size {min_size}")
    
    x1, y1, x2, y2 = box
    width, height = image_size
    
    # Check if box is too small
    if abs(x2 - x1) < min_size or abs(y2 - y1) < min_size:
        logger.warning(f"Box rejected: too small. Width: {abs(x2 - x1)}, Height: {abs(y2 - y1)}")
        return False
        
    # Check if box is out of bounds
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        logger.warning(f"Box rejected: out of bounds. Box: {box}, Image size: {image_size}")
        return False
    
    logger.debug("Box validation passed")
    return True

def create_yolo_dataset(
    output_dir: str = "Data/3_WebUI_7k/yolo_dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    sample_ratio: float = 1.0
) -> None:
    """Create YOLO format dataset from WebUI dataset.
    
    Parameters
    ----------
    output_dir : str
        Output directory for YOLO dataset
    train_ratio : float
        Ratio of training data
    val_ratio : float
        Ratio of validation data
    test_ratio : float
        Ratio of test data
    sample_ratio : float
        Ratio of the total dataset to use (default: 1.0, use 0.01 for 1% of the dataset)
    """
    logger.info("Starting YOLO dataset creation process...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    logger.info(f"Using {sample_ratio * 100}% of the total dataset")
    
    # Load dataset
    logger.info("Loading dataset from biglab/webui-7kbal-elements...")
    dataset = load_dataset("biglab/webui-7kbal-elements")
    logger.info(f"Dataset loaded successfully with {len(dataset['train'])} examples")
    
    # Sample the dataset if sample_ratio < 1
    if sample_ratio < 1.0:
        total_samples = len(dataset['train'])
        sample_size = int(total_samples * sample_ratio)
        np.random.seed(42)  # For reproducibility
        
        # Create a random sample of indices and sort them for consistency
        sampled_indices = sorted(np.random.choice(
            range(total_samples), 
            size=sample_size, 
            replace=False
        ).tolist())
        
        # Create a filtered dataset
        filtered_dataset = dataset['train'].select(sampled_indices)
        logger.info(f"Sampled {sample_size} examples from {total_samples} total examples")
    else:
        filtered_dataset = dataset['train']
    
    # Create directory structure
    output_path = Path(output_dir)
    logger.info("Creating directory structure...")
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure created successfully")
    
    # Get unique compound labels and create class mapping
    logger.info("Creating class mapping from dataset labels...")
    all_labels = set()
    
    # Use tqdm for the label collection process
    for example in tqdm(filtered_dataset, desc="Collecting unique labels", unit="example"):
        for label_list in example['labels']:
            compound_label = create_compound_label(label_list)
            all_labels.add(compound_label)
    
    class_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    logger.info(f"Found {len(class_mapping)} unique compound labels")
    for label, idx in class_mapping.items():
        logger.debug(f"Class {idx}: {label}")
    
    # Create splits
    logger.info("Creating dataset splits...")
    indices = list(range(len(filtered_dataset)))
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=test_ratio, 
        random_state=42
    )
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio/(train_ratio + val_ratio),
        random_state=42
    )
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    logger.info(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Process dataset
    total_processed = 0
    total_skipped = 0
    
    for split_name, split_indices in splits.items():
        logger.info(f"Processing {split_name} split...")
        
        processed_count = 0
        skipped_count = 0
        
        # Use tqdm for the main processing loop
        for idx in tqdm(split_indices, desc=f"Processing {split_name} split", unit="image"):
            example = filtered_dataset[idx]
            image = example['image']
            
            # Save image
            try:
                # Convert RGBA to RGB if needed
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image_filename = f"{idx:06d}.jpg"
                image_path = output_path / split_name / 'images' / image_filename
                image.save(image_path)
            except Exception as e:
                logger.error(f"Error saving image {idx}: {str(e)}")
                skipped_count += 1
                continue  # Skip this image and its annotations
            
            # Create YOLO format annotations
            label_path = output_path / split_name / 'labels' / f"{idx:06d}.txt"
            valid_annotations = 0
            
            with open(label_path, 'w') as f:
                for label_list, box in zip(example['labels'], example['contentBoxes']):
                    if not label_list:
                        logger.debug(f"Skipping empty label list for image {idx}")
                        continue
                    
                    # Skip invalid boxes
                    if not is_valid_box(box, image.size):
                        logger.debug(f"Skipping invalid box for image {idx}: {box}")
                        skipped_count += 1
                        continue
                        
                    compound_label = create_compound_label(label_list)
                    class_id = class_mapping[compound_label]
                    
                    # Convert box with proper scaling based on device
                    yolo_box = convert_box_to_yolo_format(
                        box, 
                        image.size,
                        example['key_name']
                    )
                    
                    # Write YOLO format line
                    f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
                    valid_annotations += 1
            
            processed_count += 1
        
        total_processed += processed_count
        total_skipped += skipped_count
        
        logger.info(f"Completed {split_name} split processing:")
        logger.info(f"- Processed {processed_count} images")
        logger.info(f"- Skipped {skipped_count} invalid boxes")
    
    logger.info("Dataset creation completed:")
    logger.info(f"- Total images processed: {total_processed}")
    logger.info(f"- Total boxes skipped: {total_skipped}")
    
    # Create dataset.yaml
    logger.info("Creating dataset.yaml...")
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {v: k for k, v in class_mapping.items()}
    }
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    logger.info("Dataset creation completed successfully!")

if __name__ == "__main__":
    # Create dataset with only 1% of the data for testing
    create_yolo_dataset(sample_ratio=0.1)
