from ultralytics import YOLO
import logging
from pathlib import Path
import yaml
from typing import Optional, Dict, Any
import torch
import shutil
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset_config(dataset_path: str = "Data/3_WebUI_7k/yolo_dataset") -> Dict[str, Any]:
    """Load the dataset configuration from yaml file.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory containing dataset.yaml
        
    Returns
    -------
    Dict[str, Any]
        Dataset configuration
    """
    yaml_path = Path(dataset_path) / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found at {yaml_path}")
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded dataset config with {len(config['names'])} classes")
    return config

def train_yolo(
    dataset_path: str = "Data/3_WebUI_7k/yolo_dataset",
    model_size: str = "n",  # n=nano, s=small, m=medium, l=large, x=xlarge
    pretrained: bool = True,
    epochs: int = 100,
    image_size: int = 640,
    batch_size: int = 16,
    device: Optional[str] = None
) -> YOLO:
    """Train YOLOv11 model on the dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory
    model_size : str
        Size of the YOLO model (n, s, m, l, x)
    pretrained : bool
        Whether to use pretrained weights
    epochs : int
        Number of training epochs
    image_size : int
        Input image size
    batch_size : int
        Training batch size
    device : Optional[str]
        Device to train on (cuda device, i.e. "0" or "0,1,2,3" or "cpu")
        
    Returns
    -------
    YOLO
        Trained YOLO model
    """
    logger.info("Starting YOLOv11 training process...")
    
    # Load dataset config
    dataset_config = load_dataset_config(dataset_path)
    
    # Determine model name
    model_name = f"yolo11{model_size}"
    logger.info(f"Using model: {model_name}")
    
    # Initialize model
    model = YOLO(model_name)
    
    # Set up training arguments
    train_args = {
        "data": str(Path(dataset_path) / "dataset.yaml"),
        "epochs": epochs,
        # "imgsz": image_size,
        "batch": batch_size,
        "device": device if device else ("0" if torch.cuda.is_available() else "cpu"),
        "workers": 8,
        "patience": 50,  # Early stopping patience
        "save": True,  # Save checkpoints
        "save_period": 10,  # Save every 10 epochs
        "cache": True,  # Cache images for faster training
        "exist_ok": True,  # Overwrite existing experiment
        "pretrained": pretrained,
        "optimizer": "auto",  # Use default optimizer
        "verbose": True,  # Print verbose output
        "seed": 42,  # For reproducibility
    }
    
    logger.info("Training configuration:")
    for k, v in train_args.items():
        logger.info(f"  {k}: {v}")
    
    # Train the model
    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully!")
        
        # Validate the model
        logger.info("Running validation...")
        metrics = model.val()
        
        logger.info("Validation metrics:")
        logger.info(f"  mAP50-95: {metrics.box.map:.3f}")
        logger.info(f"  mAP50: {metrics.box.map50:.3f}")
        logger.info(f"  mAP75: {metrics.box.map75:.3f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def export_model(
    model: YOLO,
    format: str = "onnx",
    output_dir: Optional[str] = None
) -> str:
    """Export the trained model to different formats.
    
    Parameters
    ----------
    model : YOLO
        Trained YOLO model
    format : str
        Export format (onnx, torchscript, openvino, etc.)
    output_dir : Optional[str]
        Directory to save the exported model
        
    Returns
    -------
    str
        Path to the exported model
    """
    try:
        logger.info(f"Exporting model to {format} format...")
        output_dir_ = model.export(format=format)
        logger.info(f"Model exported to {output_dir_}")
        
        if output_dir:
            output_dir_ = Path(output_dir) / format
            output_dir_.mkdir(parents=True, exist_ok=True)
            shutil.move(output_dir_, output_dir_)
            logger.info(f"Model moved to {output_dir_}")
        logger.info("Model exported successfully!")
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise
    return output_dir_
def benchmark_model(
    model: YOLO,
    dataset_path: str,
    image_size: int = 640,
) -> None:
    """Benchmark the trained model.
    
    Parameters
    ----------
    model : YOLO
        Trained YOLO model
    dataset_path : str
        Path to the dataset
    image_size : int
        Input image size
    """
    try:
        from ultralytics.utils.benchmarks import benchmark
        
        logger.info("Running benchmark...")
        benchmark(
            model=model,
            data=str(Path(dataset_path) / "dataset.yaml"),
            imgsz=image_size,
            half=True  # Use FP16 for faster inference
        )
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        raise

if __name__ == "__main__":
    # Train YOLOv11 model
    model = train_yolo(
        dataset_path="Data/3_WebUI_7k/yolo_dataset",
        model_size="n",  # Using medium size model
        pretrained=True,  # Use pretrained weights
        epochs=5,
        # image_size=640,
        batch_size=16
    )
    
    # Export model to ONNX format
    onnx_model_path = export_model(model, format="onnx")
    
    # Run benchmark on the trained model
    benchmark_model(
        model=model,
        dataset_path="Data/3_WebUI_7k/yolo_dataset"
    ) 
    
    # print the file path of the onnx model
    logger.info("-"*50)
    logger.info(f"ONNX model saved to {onnx_model_path}")
    logger.info("-"*50)