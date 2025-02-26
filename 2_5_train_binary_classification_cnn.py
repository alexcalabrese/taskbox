"""
Train binary classification model for lesion detection using a CNN from scratch.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import logging
from typing import Tuple, Dict, List
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
CHANNELS = 3
BATCH_SIZES_TO_TRY = [32]
LEARNING_RATE = 1e-4
EPOCHS = 40

def create_data_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = 32,
    subsample_fraction: float = 1.0
) -> Tuple[Iterator, Iterator, Iterator]:
    """Create train, validation and test data generators with balanced classes.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    val_df : pd.DataFrame
        Validation data
    img_size : Tuple[int, int]
        Image dimensions
    batch_size : int
        Batch size
    subsample_fraction : float
        Fraction of data to use (0.0 to 1.0)
        
    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        Train, validation and test data generators
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
    
    # Balance classes by oversampling minority class
    def balance_dataset(df):
        df = df.copy()
        # Convert boolean to string
        df['is_lesion'] = df['is_lesion'].map({True: 'lesion', False: 'no_lesion'})
        
        lesion_df = df[df['is_lesion'] == 'lesion']
        no_lesion_df = df[df['is_lesion'] == 'no_lesion']
        
        if len(lesion_df) > len(no_lesion_df):
            no_lesion_df = no_lesion_df.sample(n=len(lesion_df), replace=True, random_state=42)
            return pd.concat([lesion_df, no_lesion_df])
        else:
            lesion_df = lesion_df.sample(n=len(no_lesion_df), replace=True, random_state=42)
            return pd.concat([no_lesion_df, lesion_df])
    
    train_df_balanced = balance_dataset(train_df)
    
    # Convert validation and test sets
    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df['is_lesion'] = val_df['is_lesion'].map({True: 'lesion', False: 'no_lesion'})
    test_df['is_lesion'] = test_df['is_lesion'].map({True: 'lesion', False: 'no_lesion'})
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df_balanced,
        x_col='frame_path',
        y_col='is_lesion',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['no_lesion', 'lesion'],
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='frame_path',
        y_col='is_lesion',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['no_lesion', 'lesion'],
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='frame_path',
        y_col='is_lesion',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=['no_lesion', 'lesion'],
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Create CNN model from scratch.
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Input image shape (height, width, channels)
        
    Returns
    -------
    tf.keras.Model
        Compiled model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score')
        ]
    )
    
    return model

def train_model_with_batch_size(
    train_generator: Iterator,
    val_generator: Iterator,
    batch_size: int,
    num_epochs: int = EPOCHS
) -> Tuple[tf.keras.Model, Dict, float]:
    """Train model with specific batch size.
    
    Parameters
    ----------
    train_generator : Iterator
        Training data generator
    val_generator : Iterator
        Validation data generator
    batch_size : int
        Batch size to try
    num_epochs : int
        Number of epochs to train
        
    Returns
    -------
    Tuple[tf.keras.Model, Dict, float]
        Trained model, training history, and best validation accuracy
    """
    logger.info(f"\nTraining with batch size = {batch_size}")
    
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        callbacks=[early_stop],
        verbose=1
    )
    
    val_acc = max(history.history['val_accuracy'])
    
    return model, history.history, val_acc

def plot_training_history(history: Dict, save_path: Path):
    """Plot and save training history."""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(history['f1_score'], label='Training')
    plt.plot(history['val_f1_score'], label='Validation')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()

def evaluate_model(
    model: tf.keras.Model,
    test_generator: Iterator,
    save_path: Path
):
    """Evaluate model and save detailed results."""
    # Get predictions
    predictions = model.predict(test_generator)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Get true classes - test_generator.classes already contains the correct indices
    true_classes = test_generator.classes
    
    # Calculate metrics
    cm = confusion_matrix(true_classes, pred_classes)
    cr = classification_report(
        true_classes,
        pred_classes,
        target_names=['No Lesion', 'Lesion'],
        digits=4
    )
    
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

def main():
    parser = argparse.ArgumentParser(description='Train CNN binary classification model')
    parser.add_argument('--data-path', type=str, 
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--output-path', type=str,
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/models/cnn_scratch",
                       help='Path to save model and results')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Fraction of dataset to use (0.0 to 1.0)')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_and_filter_data(Path(args.data_path))
    
    # Split into train, validation, and test
    train_val_df, test_df = create_patient_wise_split(df, test_size=0.2)
    train_df, val_df = create_patient_wise_split(train_val_df, test_size=0.2)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Try different batch sizes
    best_val_acc = 0.0
    best_batch_size = None
    best_model = None
    best_history = None
    
    for batch_size in BATCH_SIZES_TO_TRY:
        # Create data generators for this batch size
        train_generator, val_generator, test_generator = create_data_generators(
            train_df, test_df, val_df,
            batch_size=batch_size,
            subsample_fraction=args.subsample
        )
        
        # Train model with this batch size
        model, history, val_acc = train_model_with_batch_size(
            train_generator,
            val_generator,
            batch_size,
            num_epochs=args.epochs
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_batch_size = batch_size
            best_model = model
            best_history = history
    
    logger.info(f"\nBest batch size found: {best_batch_size}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history for best model
    plot_training_history(best_history, output_path)
    
    # Create final test generator with best batch size
    _, _, test_generator = create_data_generators(
        train_df, test_df, val_df,
        batch_size=best_batch_size,
        subsample_fraction=args.subsample
    )
    
    # Evaluate best model
    logger.info("\nEvaluating best model...")
    try:
        evaluate_model(best_model, test_generator, output_path)
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
    
    model_name = f"model_batch_size_{best_batch_size}_subsample_{args.subsample}.h5"
    # Save best model
    best_model.save(output_path / model_name)
    logger.info(f"Best model saved to {output_path / model_name}")
    print(f"Model total params: {best_model.count_params()}")
if __name__ == "__main__":
    main()

# usage:
# python 2_5_train_binary_classification_cnn.py  # default settings
# python 2_5_train_binary_classification_cnn.py --epochs 100  # train for more epochs
# python 2_5_train_binary_classification_cnn.py --subsample 0.1  # use 10% of data 