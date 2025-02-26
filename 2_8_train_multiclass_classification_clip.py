"""
Train multilabel classification model using CLIP embeddings.
Adapts CLIP for medical image classification with multiple lesion types.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from typing import Tuple, Dict, List
import argparse
import importlib
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel
from PIL import Image

prepare_multiclass = importlib.import_module("2_6_prepare_multiclass_classification")
load_and_filter_data = prepare_multiclass.load_and_filter_data
create_patient_wise_split = prepare_multiclass.create_patient_wise_split
parse_annotations = prepare_multiclass.parse_annotations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 40

class CADICADataset(Dataset):
    """Custom Dataset for CADICA frames with multilabel annotations."""
    
    def __init__(self, dataframe: pd.DataFrame, mlb: MultiLabelBinarizer, transform=None):
        self.dataframe = dataframe
        self.mlb = mlb
        self.transform = transform
        self.labels = self._extract_labels()
        
    def _extract_labels(self) -> np.ndarray:
        """Extract and binarize labels from annotations."""
        label_lists = []
        for annotations in self.dataframe['parsed_annotations']:
            frame_labels = set()
            for ann in annotations:
                if 'class' in ann:
                    frame_labels.add(ann['class'])
            label_lists.append(list(frame_labels))
        return self.mlb.transform(label_lists)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['frame_path']
        label = self.labels[idx]
        
        # Load image
        image = plt.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    batch_size: int = BATCH_SIZE,
    subsample_fraction: float = 1.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test data loaders.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    mlb : MultiLabelBinarizer
        Fitted label binarizer
    batch_size : int
        Batch size
    subsample_fraction : float
        Fraction of data to use (0.0 to 1.0)
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        Train, validation and test data loaders
    """
    # Subsample if requested
    if subsample_fraction < 1.0:
        train_df = train_df.sample(frac=subsample_fraction, random_state=42)
        val_df = val_df.sample(frac=subsample_fraction, random_state=42)
        test_df = test_df.sample(frac=subsample_fraction, random_state=42)
        logger.info(f"Subsampled to {subsample_fraction:.1%} of data:")
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        logger.info(f"Testing samples: {len(test_df)}")

    # --- Begin Added Balancing Code for Training Set ---
    train_df = train_df.copy()
    train_df['primary_class'] = train_df['parsed_annotations'].apply(lambda anns: anns[0].get('class') if (len(anns) > 0 and 'class' in anns[0]) else None)
    class_counts = train_df['primary_class'].value_counts()
    max_count = class_counts.max()
    balanced_dfs = []
    for cls, count in class_counts.items():
        cls_df = train_df[train_df['primary_class'] == cls]
        if count < max_count:
            cls_df = cls_df.sample(n=max_count, replace=True, random_state=42)
            logger.info(f"Balanced {cls} class: {count} -> {max_count}")
        balanced_dfs.append(cls_df)
    train_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_df.drop(columns=['primary_class'])
    # --- End Added Balancing Code ---
    
    # Modified transform for grayscale images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485],  # Single mean value for grayscale
            std=[0.229]    # Single std value for grayscale
        )
    ])
    
    train_dataset = CADICADataset(train_df, mlb, transform=transform)
    val_dataset = CADICADataset(val_df, mlb, transform=transform)
    test_dataset = CADICADataset(test_df, mlb, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class CLIPMultilabelClassifier(torch.nn.Module):
    """CLIP-based multilabel classifier."""
    
    def __init__(self, num_classes: int):
        super(CLIPMultilabelClassifier, self).__init__()
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.vision.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
            torch.nn.Sigmoid()
        )
        
    def forward(self, pixel_values):
        outputs = self.vision(pixel_values=pixel_values)
        image_embeds = outputs.pooler_output
        logits = self.classifier(image_embeds)
        return logits

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: Path,
    learning_rate: float = LEARNING_RATE
) -> Dict:
    """Train the model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int
        Number of epochs to train
    device : torch.device
        Device to train on
    save_path : Path
        Path to save model checkpoints
    learning_rate : float
        Learning rate for optimizer
        
    Returns
    -------
    Dict
        Training history
    """
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy for multilabel
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_total += labels.size(0) * labels.size(1)  # Total number of predictions
            train_correct += (predictions == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_total += labels.size(0) * labels.size(1)
                val_correct += (predictions == labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Log epoch results
        logger.info(
            f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
        )
        logger.info(
            f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path / 'best_model.pth')
            logger.info("Best model updated!")
    
    return history

def plot_training_history(history: Dict, save_path: Path):
    """Plot and save training history."""
    metrics_to_plot = [
        ('acc', 'Accuracy'),
        ('loss', 'Loss')
    ]
    
    plt.figure(figsize=(12, 5))
    
    for i, (metric, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 2, i)
        plt.plot(history[f'train_{metric}'], label='Training')
        plt.plot(history[f'val_{metric}'], label='Validation')
        plt.title(f'Model {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    mlb: MultiLabelBinarizer,
    device: torch.device,
    save_path: Path,
    threshold: float = 0.5
):
    """Evaluate model and save detailed results."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(pixel_values=images)
            predictions = (outputs > threshold).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate per-class metrics
    class_names = mlb.classes_
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        tn = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 0))
        fp = np.sum((all_labels[:, i] == 0) & (all_preds[:, i] == 1))
        fn = np.sum((all_labels[:, i] == 1) & (all_preds[:, i] == 0))
        tp = np.sum((all_labels[:, i] == 1) & (all_preds[:, i] == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calculate F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1,
            'support': np.sum(all_labels[:, i])
        }
    
    # Calculate weighted accuracy
    total_samples = np.sum([metrics['support'] for metrics in class_metrics.values()])
    weighted_acc = np.sum([
        metrics['sensitivity'] * metrics['support'] 
        for metrics in class_metrics.values()
    ]) / total_samples
    
    # Save detailed results
    with open(save_path / 'evaluation_results.txt', 'w') as f:
        f.write("=== Model Evaluation Results ===\n\n")
        f.write(f"Overall Weighted Accuracy: {weighted_acc:.4f}\n\n")
        
        for class_name, metrics in class_metrics.items():
            f.write(f"\nClass: {class_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Support: {metrics['support']}\n")
            f.write(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"PPV (Precision): {metrics['ppv']:.4f}\n")
            f.write(f"NPV: {metrics['npv']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
    
    # Log results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"\nOverall Weighted Accuracy: {weighted_acc:.4f}")
    for class_name, metrics in class_metrics.items():
        logger.info(f"\nClass: {class_name}")
        logger.info(f"Support: {metrics['support']}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"PPV: {metrics['ppv']:.4f}")
        logger.info(f"NPV: {metrics['npv']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrices for each class
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}\nF1={class_metrics[class_name]["f1_score"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path / f'confusion_matrix_{class_name}.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train multilabel CLIP classification model')
    parser.add_argument('--merge-p99-p100', dest='merge_p99_p100', action='store_true', help="Merge classes 'p99' and 'p100' into a single class 'p99_p100'")
    parser.set_defaults(merge_p99_p100=True)
    parser.add_argument('--data-path', type=str, 
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--output-path', type=str,
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/models/multilabel_clip",
                       help='Path to save model and results')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0 to 1.0)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_and_filter_data(Path(args.data_path))
    
    # If merge_p99_p100 flag is set, merge classes 'p99' and 'p100' into 'p99_p100'
    if args.merge_p99_p100:
        logger.info("Merging classes p99 and p100 into 'p99_p100'")
        def merge_classes(anns: list) -> list:
            for ann in anns:
                if 'class' in ann and ann['class'] in ['p99', 'p100']:
                    ann['class'] = 'p99_p100'
            return anns
        df['parsed_annotations'] = df['parsed_annotations'].apply(merge_classes)
    
    # Split into train, validation, and test
    train_val_df, test_df = create_patient_wise_split(df, test_size=0.2)
    train_df, val_df = create_patient_wise_split(train_val_df, test_size=0.2)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Initialize label binarizer with all classes
    all_classes = set()
    for annotations in df['parsed_annotations']:
        for ann in annotations:
            if 'class' in ann:
                all_classes.add(ann['class'])
    mlb = MultiLabelBinarizer(classes=sorted(list(all_classes)))
    mlb.fit([[]])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        mlb=mlb,
        batch_size=args.batch_size,
        subsample_fraction=args.subsample
    )
    
    # Create and train model
    logger.info("Creating and training model...")
    model = CLIPMultilabelClassifier(num_classes=len(mlb.classes_)).to(device)
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=device,
        save_path=output_path
    )
    
    # Plot training history
    plot_training_history(history, output_path)
    
    # Evaluate model
    logger.info("\nEvaluating model...")
    evaluate_model(model, test_loader, mlb, device, output_path)

if __name__ == "__main__":
    main()

# usage:
# python 2_8_train_multiclass_classification_clip.py  # default settings
# python 2_8_train_multiclass_classification_clip.py --epochs 100  # train for more epochs
# python 2_8_train_multiclass_classification_clip.py --subsample 0.1  # use 10% of data 