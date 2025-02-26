"""
Train binary classification model for lesion detection using EfficientNetV2.
"""

from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report
import logging
from typing import Tuple, Dict, Any
import argparse
import importlib
import torch

prepare_binary = importlib.import_module("2_2_prepare_binary_classification")
load_and_filter_data = prepare_binary.load_and_filter_data
create_patient_wise_split = prepare_binary.create_patient_wise_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
CHANNELS = 3

def create_data_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    subsample_fraction: float = 1.0
) -> Tuple[Iterator, Iterator]:
    """Create train and test data generators with balanced classes.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    img_size : Tuple[int, int]
        Image dimensions
    batch_size : int
        Batch size
    subsample_fraction : float
        Fraction of data to use (0.0 to 1.0), default=1.0 uses all data
        
    Returns
    -------
    Tuple[Iterator, Iterator]
        Train and test data generators
    """
    # Subsample the data if requested
    if subsample_fraction < 1.0:
        train_df = train_df.sample(frac=subsample_fraction, random_state=42)
        test_df = test_df.sample(frac=subsample_fraction, random_state=42)
        logger.info(f"Subsampled to {subsample_fraction:.1%} of data:")
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Testing samples: {len(test_df)}")
    
    # Convert boolean to string labels
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['is_lesion'] = train_df['is_lesion'].map({True: 'lesion', False: 'no_lesion'})
    test_df['is_lesion'] = test_df['is_lesion'].map({True: 'lesion', False: 'no_lesion'})
    
    # Balance classes by oversampling minority class
    lesion_df = train_df[train_df['is_lesion'] == 'lesion']
    no_lesion_df = train_df[train_df['is_lesion'] == 'no_lesion']
    
    # Calculate how many samples we need to add
    n_no_lesion = len(no_lesion_df)
    n_lesion = len(lesion_df)
    
    # Oversample the minority class to match the majority class
    if n_lesion > n_no_lesion:
        # No lesion is minority class
        no_lesion_df_oversampled = no_lesion_df.sample(n=n_lesion, replace=True, random_state=42)
        train_df_balanced = pd.concat([lesion_df, no_lesion_df_oversampled])
    else:
        # Lesion is minority class
        lesion_df_oversampled = lesion_df.sample(n=n_no_lesion, replace=True, random_state=42)
        train_df_balanced = pd.concat([no_lesion_df, lesion_df_oversampled])
    
    # Shuffle the balanced dataset
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Original class distribution - No Lesion: {n_no_lesion}, Lesion: {n_lesion}")
    logger.info(f"Balanced class distribution - No Lesion: {len(train_df_balanced[train_df_balanced['is_lesion'] == 'no_lesion'])}, "
                f"Lesion: {len(train_df_balanced[train_df_balanced['is_lesion'] == 'lesion'])}")
    
    # pixel value before rescale
    logger.info(f"Pixel value before rescale 1./255 : {train_df_balanced['frame_path'].iloc[0]}")
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        # Add more augmentation for better generalization
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        shear_range=0.1
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_dataframe(
        train_df_balanced,  # Use balanced dataset
        x_col='frame_path',
        y_col='is_lesion',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col='frame_path',
        y_col='is_lesion',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, test_gen

def create_model(img_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Create EfficientNetV2 based model for binary classification.
    
    Parameters
    ----------
    img_shape : Tuple[int, int, int]
        Input image shape (height, width, channels)
        
    Returns
    -------
    tf.keras.Model
        Compiled model
    """
    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
        pooling='max'
    )
    
    # Unfreeze the base model for fine-tuning
    base_model.trainable = True
    
    # Use a lower learning rate for fine-tuning
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006),
              activation='relu'),
        Dropout(rate=0.45),
        Dense(128, activation='relu'),
        Dropout(rate=0.35),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adamax(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(name='precision_no_lesion', class_id=0),
            tf.keras.metrics.Recall(name='recall_no_lesion', class_id=0),
            tf.keras.metrics.F1Score(name='f1_score', average=None),
            tf.keras.metrics.Precision(name='precision_lesion', class_id=1),
            tf.keras.metrics.Recall(name='recall_lesion', class_id=1)
        ]
    )
    
    return model

def plot_training_history(history: tf.keras.callbacks.History, save_path: Path):
    """Plot and save training history.
    
    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history
    save_path : Path
        Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()

def evaluate_model(model: tf.keras.Model, 
                  test_gen: Iterator,
                  save_path: Path):
    """Evaluate model and save results.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    test_gen : Iterator
        Test data generator
    save_path : Path
        Path to save results
    """
    # Get predictions
    predictions = model.predict(test_gen)
    pred_classes = np.argmax(predictions, axis=1)  # Get class with highest probability
    true_classes = test_gen.classes
    
    # Calculate metrics
    cm = confusion_matrix(true_classes, pred_classes)
    cr = classification_report(true_classes, pred_classes,
                             target_names=['No Lesion', 'Lesion'])
    
    # Save results
    with open(save_path / 'evaluation_results.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(cr)
    
    logger.info("Evaluation Results:")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{cr}")

def main():
    parser = argparse.ArgumentParser(description='Train binary classification model')
    parser.add_argument('--data-path', type=str, 
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--output-path', type=str,
                       default="/teamspace/studios/this_studio/dsim/Data/2_CADICA/models",
                       help='Path to save model and results')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_and_filter_data(Path(args.data_path))
    train_df, test_df = create_patient_wise_split(df)
    
    # Create data generators
    train_gen, test_gen = create_data_generators(train_df, test_df, subsample_fraction=0.1)
    
    # Create and train model
    logger.info("Creating and training model...")
    model = create_model(img_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=test_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Plot training history
    plot_training_history(history, output_path)
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        evaluate_model(model, test_gen, output_path)
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lesion_detection_efficientnetv2m_subsample_100p_epochs_{args.epochs}p_weighted_{timestamp}.h5"
    # Save model
    model.save(output_path / model_name)
    logger.info(f"Model saved to {output_path / model_name}")
    print(f"Model total params: {model.count_params()}")

if __name__ == "__main__":
    main()

# usage:
# python 2_3_train_binary_classification.py  # default settings
# python 2_3_train_binary_classification.py --epochs 100  # train for more epochs
# python 2_3_train_binary_classification.py --output-path "/custom/path"  # custom save location 