from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
import random
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_class_names(dataset_yaml: str = "Data/3_WebUI_7k/yolo_dataset/dataset.yaml") -> Dict[int, str]:
    """Load class names from dataset yaml file.
    
    Parameters
    ----------
    dataset_yaml : str
        Path to dataset yaml file
        
    Returns
    -------
    Dict[int, str]
        Mapping of class indices to names
    """
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def get_random_test_images(
    test_dir: str = "Data/3_WebUI_7k/yolo_dataset/test/images",
    num_samples: int = 5
) -> List[Tuple[str, str]]:
    """Get random test images and their corresponding label files.
    
    Parameters
    ----------
    test_dir : str
        Directory containing test images
    num_samples : int
        Number of random samples to select
        
    Returns
    -------
    List[Tuple[str, str]]
        List of (image_path, label_path) tuples
    """
    image_dir = Path(test_dir)
    label_dir = image_dir.parent.parent / "test" / "labels"
    
    image_files = list(image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {test_dir}")
    
    selected = random.sample(image_files, min(num_samples, len(image_files)))
    return [(str(img), str(label_dir / img.stem) + ".txt") for img in selected]

def read_true_labels(label_path: str, class_names: Dict[int, str]) -> List[Tuple[str, List[float]]]:
    """Read true labels from a label file.
    
    Parameters
    ----------
    label_path : str
        Path to label file
    class_names : Dict[int, str]
        Mapping of class indices to names
        
    Returns
    -------
    List[Tuple[str, List[float]]]
        List of (class_name, bbox) tuples
    """
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class_id x_center y_center width height
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    class_name = class_names.get(class_id, f"unknown_{class_id}")
                    labels.append((class_name, bbox))
    except Exception as e:
        logger.error(f"Error reading label file {label_path}: {str(e)}")
    return labels

def visualize_results(
    image_path: str,
    true_labels: List[Tuple[str, List[float]]],
    pred_results,
    class_names: Dict[int, str],
    output_dir: str,
    conf_threshold: float = 0.25
) -> str:
    """Visualize true and predicted labels on the image.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    true_labels : List[Tuple[str, List[float]]]
        List of true (class_name, bbox) tuples
    pred_results : ultralytics.engine.results.Results
        Prediction results from YOLO model
    class_names : Dict[int, str]
        Mapping of class indices to names
    output_dir : str
        Directory to save visualization
    conf_threshold : float
        Confidence threshold for predictions
        
    Returns
    -------
    str
        Path to saved visualization
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original image with true labels
    ax1.imshow(img)
    ax1.set_title("Ground Truth")
    for class_name, bbox in true_labels:
        # Convert YOLO format to pixel coordinates
        x_center, y_center, w, h = bbox
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
        ax1.add_patch(rect)
        # Add label
        ax1.text(x1, y1-5, class_name, color='green', fontsize=8, backgroundcolor='white')
    
    # Plot original image with predictions
    ax2.imshow(img)
    ax2.set_title("Predictions")
    
    # Draw predicted boxes
    for box in pred_results.boxes:
        if box.conf.item() < conf_threshold:
            continue
            
        # Get coordinates and class
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cls = int(box.cls.item())
        class_name = class_names.get(cls, f"unknown_{cls}")
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax2.add_patch(rect)
        # Add label with confidence
        ax2.text(x1, y1-5, f"{class_name} {conf:.2f}", color='red', fontsize=8, backgroundcolor='white')
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Save figure
    output_path = Path(output_dir) / f"{Path(image_path).stem}_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return str(output_path)

def run_inference(
    model_path: str = "runs/detect/train/weights/best.onnx",
    dataset_yaml: str = "Data/3_WebUI_7k/yolo_dataset/dataset.yaml",
    num_samples: int = 5,
    conf_threshold: float = 0.25
) -> None:
    """Run inference on random test images and visualize results.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model
    dataset_yaml : str
        Path to dataset yaml file
    num_samples : int
        Number of random samples to process
    conf_threshold : float
        Confidence threshold for predictions
    """
    logger.info("Starting inference process...")
    
    # Statistics tracking
    processed_count = 0
    skipped_count = 0
    error_details = []
    
    try:
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"inference_results_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Load model and class names
        try:
            model = YOLO(model_path)
            class_names = load_class_names(dataset_yaml)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model or class names: {str(e)}")
            raise
        
        # Get random test images
        try:
            test_samples = get_random_test_images(num_samples=num_samples)
            logger.info(f"Selected {len(test_samples)} random test samples")
        except Exception as e:
            logger.error(f"Failed to get test samples: {str(e)}")
            raise
        
        # Process each sample
        for image_path, label_path in test_samples:
            try:
                logger.info(f"Processing {image_path}")
                
                # Check if image exists
                if not Path(image_path).exists():
                    logger.warning(f"Image not found: {image_path}")
                    skipped_count += 1
                    error_details.append(f"Image not found: {image_path}")
                    continue
                
                # Check if label file exists
                if not Path(label_path).exists():
                    logger.warning(f"Label file not found: {label_path}")
                    skipped_count += 1
                    error_details.append(f"Label file not found: {label_path}")
                    continue
                
                # Get true labels
                true_labels = read_true_labels(label_path, class_names)
                if not true_labels:
                    logger.warning(f"No valid labels found in {label_path}")
                    skipped_count += 1
                    error_details.append(f"No valid labels: {label_path}")
                    continue
                
                # Run inference
                try:
                    results = model(image_path)[0]
                except Exception as e:
                    logger.error(f"Inference failed for {image_path}: {str(e)}")
                    skipped_count += 1
                    error_details.append(f"Inference failed: {image_path} - {str(e)}")
                    continue
                
                # Visualize and save results
                try:
                    output_path = visualize_results(
                        image_path,
                        true_labels,
                        results,
                        class_names,
                        output_dir,
                        conf_threshold
                    )
                    logger.info(f"Saved visualization to {output_path}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Visualization failed for {image_path}: {str(e)}")
                    skipped_count += 1
                    error_details.append(f"Visualization failed: {image_path} - {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                skipped_count += 1
                error_details.append(f"Processing failed: {image_path} - {str(e)}")
                continue
        
        # Save error report
        if error_details:
            error_report_path = output_dir / "error_report.txt"
            with open(error_report_path, 'w') as f:
                f.write("Error Report\n")
                f.write("="*50 + "\n")
                f.write(f"Total processed: {processed_count}\n")
                f.write(f"Total skipped: {skipped_count}\n")
                f.write("="*50 + "\n")
                f.write("\nDetailed Errors:\n")
                for error in error_details:
                    f.write(f"- {error}\n")
        
        # Log final statistics
        logger.info("="*50)
        logger.info("Inference completed:")
        logger.info(f"- Total images processed successfully: {processed_count}")
        logger.info(f"- Total images skipped: {skipped_count}")
        if error_details:
            logger.info(f"- Error details saved to: {error_report_path}")
        logger.info(f"- Results saved in: {output_dir}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error during inference process: {str(e)}")
        raise

if __name__ == "__main__":
    run_inference(
        # model_path="runs/detect/train/weights/best.onnx",
        model_path="/teamspace/studios/this_studio/dsim/best.pt",
        num_samples=5,
        conf_threshold=0.25
    ) 