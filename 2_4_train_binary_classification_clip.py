"""
Train binary classification model for lesion detection using CLIP Vision model.
"""

from datetime import datetime
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel
from sklearn.metrics import confusion_matrix, classification_report
import logging
from typing import Tuple, Dict
import argparse
import importlib
import seaborn as sns
from tqdm import tqdm

prepare_binary = importlib.import_module("2_2_prepare_binary_classification")
load_and_filter_data = prepare_binary.load_and_filter_data
create_patient_wise_split = prepare_binary.create_patient_wise_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 5e-5

class CADICADataset(Dataset):
    """Custom Dataset for CADICA frames."""
    
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['frame_path']
        label = self.dataframe.iloc[idx]['is_lesion']
        
        # Load image
        image = plt.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(int(label))

def create_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    subsample_fraction: float = 1.0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    batch_size : int
        Batch size
    subsample_fraction : float
        Fraction of data to use (0.0 to 1.0)
        
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Train and test data loaders
    """
    # Subsample if requested
    if subsample_fraction < 1.0:
        train_df = train_df.sample(frac=subsample_fraction, random_state=42)
        test_df = test_df.sample(frac=subsample_fraction, random_state=42)
        logger.info(f"Subsampled to {subsample_fraction:.1%} of data:")
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Testing samples: {len(test_df)}")
    
    # Balance training set by oversampling minority class
    train_df = train_df.copy()
    lesion_df = train_df[train_df['is_lesion'] == True]
    no_lesion_df = train_df[train_df['is_lesion'] == False]
    
    if len(lesion_df) > len(no_lesion_df):
        no_lesion_df = no_lesion_df.sample(n=len(lesion_df), replace=True, random_state=42)
        train_df = pd.concat([lesion_df, no_lesion_df])
    else:
        lesion_df = lesion_df.sample(n=len(no_lesion_df), replace=True, random_state=42)
        train_df = pd.concat([no_lesion_df, lesion_df])
    
    logger.info(f"Balanced training set size: {len(train_df)}")
    logger.info(f"Lesion samples: {len(train_df[train_df['is_lesion'] == True])}")
    logger.info(f"Non-lesion samples: {len(train_df[train_df['is_lesion'] == False])}")
    
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
    
    train_dataset = CADICADataset(train_df, transform=transform)
    test_dataset = CADICADataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CLIPClassifier(torch.nn.Module):
    """CLIP-based binary classifier."""
    
    def __init__(self, num_classes: int = 2):
        super(CLIPClassifier, self).__init__()
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = torch.nn.Linear(self.vision.config.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.vision(pixel_values=pixel_values)
        image_embeds = outputs.pooler_output
        logits = self.classifier(image_embeds)
        return logits

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
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
    test_loader : DataLoader
        Test/validation data loader
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc='Training', 
                         leave=False, unit='batch')
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
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
        
        val_pbar = tqdm(test_loader, desc='Validation', 
                       leave=False, unit='batch')
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(test_loader)
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
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path / 'best_model.pth')
            logger.info("Best model updated!")
    
    return history

def plot_training_history(history: Dict, save_path: Path):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: Path
):
    """Evaluate model and save detailed results.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run evaluation on
    save_path : Path
        Path to save results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For storing prediction probabilities
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(pixel_values=images)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, 
                             target_names=['No Lesion', 'Lesion'],
                             digits=4)  # More decimal places
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Save detailed results
    with open(save_path / 'evaluation_results.txt', 'w') as f:
        f.write("=== Model Evaluation Results ===\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n\n")
        
        f.write("Detailed Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Positive Predictive Value (Precision): {ppv:.4f}\n")
        f.write(f"Negative Predictive Value: {npv:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 40 + "\n")
        f.write(cr)
    
    # Log results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nSensitivity: {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"PPV: {ppv:.4f}")
    logger.info(f"NPV: {npv:.4f}")
    logger.info(f"\nClassification Report:\n{cr}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Lesion', 'Lesion'],
                yticklabels=['No Lesion', 'Lesion'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

    return {
        'confusion_matrix': cm,
        'classification_report': cr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

def main():
    parser = argparse.ArgumentParser(description='Train CLIP binary classification model')
    parser.add_argument('--data-path', type=str, 
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--output-path', type=str,
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/models",
                       help='Path to save model and results')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
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
    train_df, test_df = create_patient_wise_split(df)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_df, test_df, 
        subsample_fraction=args.subsample
    )
    
    # Create and train model
    logger.info("Creating and training model...")
    model = CLIPClassifier(num_classes=2).to(device)
    
    history = train_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs,
        device=device,
        save_path=output_path
    )
    
    # Plot training history
    plot_training_history(history, output_path)
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        evaluate_model(model, test_loader, device, output_path)
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
    

if __name__ == "__main__":
    main()

# usage:
# python 2_4_train_binary_classification_clip.py  # default settings
# python 2_4_train_binary_classification_clip.py --epochs 20  # more epochs
# python 2_4_train_binary_classification_clip.py --subsample 0.1  # use 10% of data 