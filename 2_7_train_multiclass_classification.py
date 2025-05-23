"""
Train multilabel classification model for lesion type detection using EfficientNetV2.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from typing import Tuple, Dict, List
import argparse
import importlib
import seaborn as sns
from tqdm import tqdm

prepare_multiclass = importlib.import_module("2_6_prepare_multiclass_classification")
load_and_filter_data = prepare_multiclass.load_and_filter_data
create_patient_wise_split = prepare_multiclass.create_patient_wise_split
parse_annotations = prepare_multiclass.parse_annotations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 40

class MultiLabelGenerator(Sequence):
    """Custom generator for multilabel data."""
    
    def __init__(self, dataframe: pd.DataFrame, mlb: MultiLabelBinarizer,
                 image_generator: ImageDataGenerator, **kwargs):
        """Initialize the generator.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing image paths and annotations
        mlb : MultiLabelBinarizer
            Fitted label binarizer for converting string labels to binary vectors
        image_generator : ImageDataGenerator
            Keras ImageDataGenerator for data augmentation
        **kwargs : dict
            Additional arguments including batch_size, target_size, and shuffle
        """
        self.dataframe = dataframe
        self.mlb = mlb
        self.image_generator = image_generator
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_size = kwargs.get('target_size', (224, 224))
        self.shuffle = kwargs.get('shuffle', True)
        
        # Extract labels from annotations
        self.labels = self._extract_labels()
        self.n_samples = len(self.dataframe)
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
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
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at position idx.
        
        Parameters
        ----------
        idx : int
            Position of the batch in the Sequence
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple (x, y) where x is a batch of images and y is their labels
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load and preprocess images
        batch_images = []
        for idx in batch_indices:
            img_path = self.dataframe.iloc[idx]['frame_path']
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=self.target_size
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.image_generator.standardize(img)
            batch_images.append(img)
        
        batch_x = np.array(batch_images)
        batch_y = self.labels[batch_indices]
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_data_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    subsample_fraction: float = 1.0
) -> Tuple[MultiLabelGenerator, MultiLabelGenerator, MultiLabelGenerator, MultiLabelBinarizer]:
    """Create train, validation and test data generators.
    
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
    Tuple[MultiLabelGenerator, MultiLabelGenerator, MultiLabelGenerator, MultiLabelBinarizer]
        Train, validation and test data generators, and the label binarizer
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

    # Balance the training set by upsampling based on a primary class extracted from parsed_annotations
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
    
    # Extract all unique classes
    all_classes = set()
    for df in [train_df, val_df, test_df]:
        for annotations in df['parsed_annotations']:
            for ann in annotations:
                if 'class' in ann:
                    all_classes.add(ann['class'])
    
    # Initialize the MultiLabelBinarizer with all classes
    mlb = MultiLabelBinarizer(classes=sorted(list(all_classes)))
    mlb.fit([[]])  # Fit with empty list to initialize the classes
    
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
    
    # Create generators
    train_generator = MultiLabelGenerator(
        train_df, mlb, train_datagen,
        batch_size=batch_size,
        target_size=img_size,
        shuffle=True
    )
    
    val_generator = MultiLabelGenerator(
        val_df, mlb, val_test_datagen,
        batch_size=batch_size,
        target_size=img_size,
        shuffle=False
    )
    
    test_generator = MultiLabelGenerator(
        test_df, mlb, val_test_datagen,
        batch_size=batch_size,
        target_size=img_size,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, mlb

def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Create EfficientNetV2 based model for multilabel classification.
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Input image shape (height, width, channels)
    num_classes : int
        Number of classes to predict
        
    Returns
    -------
    tf.keras.Model
        Compiled model
    """
    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='max'
    )
    
    # Unfreeze the base model for fine-tuning
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.016),
                    activity_regularizer=tf.keras.regularizers.l1(0.006),
                    bias_regularizer=tf.keras.regularizers.l1(0.006),
                    activation='relu'),
        layers.Dropout(0.45),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multilabel
    ])
    
    # Custom F1 metric for each class
    class_f1_metrics = []
    for i in range(num_classes):
        f1_metric = tf.keras.metrics.F1Score(
            name=f'f1_class_{i}',
            threshold=0.5,
            average=None
        )
        class_f1_metrics.append(f1_metric)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',  # Binary cross-entropy for multilabel
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(multi_label=True, name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            # Add weighted metrics
            tf.keras.metrics.Accuracy(name='weighted_accuracy', dtype=tf.float32),
            tf.keras.metrics.F1Score(name='weighted_f1', average='weighted'),
            # Add per-class F1 scores
            *class_f1_metrics
        ]
    )
    
    return model

def train_model(
    model: tf.keras.Model,
    train_generator: MultiLabelGenerator,
    val_generator: MultiLabelGenerator,
    num_epochs: int = EPOCHS
) -> Tuple[tf.keras.Model, Dict]:
    """Train the model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to train
    train_generator : MultiLabelGenerator
        Training data generator
    val_generator : MultiLabelGenerator
        Validation data generator
    num_epochs : int
        Number of epochs to train
        
    Returns
    -------
    Tuple[tf.keras.Model, Dict]
        Trained model and training history
    """
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
    
    return model, history.history

def plot_training_history(history: Dict, save_path: Path):
    """Plot and save training history."""
    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('loss', 'Loss'),
        ('auc', 'AUC'),
        ('weighted_accuracy', 'Weighted Accuracy'),
        ('weighted_f1', 'Weighted F1')
    ]
    
    plt.figure(figsize=(20, 5))
    
    for i, (metric, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, len(metrics_to_plot), i)
        plt.plot(history[metric], label='Training')
        plt.plot(history[f'val_{metric}'], label='Validation')
        plt.title(f'Model {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()
    
    # Plot per-class F1 scores
    f1_metrics = [m for m in history.keys() if m.startswith('f1_class_') and not m.startswith('val_')]
    if f1_metrics:
        plt.figure(figsize=(15, 5))
        for metric in f1_metrics:
            plt.plot(history[metric], label=f'Train {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Val {metric}')
        plt.title('Per-class F1 Scores')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path / 'per_class_f1_history.png')
        plt.close()

def evaluate_model(
    model: tf.keras.Model,
    test_generator: MultiLabelGenerator,
    mlb: MultiLabelBinarizer,
    save_path: Path,
    threshold: float = 0.5
):
    """Evaluate model and save detailed results."""
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Get true labels
    y_true = test_generator.labels
    
    # Calculate per-class metrics
    class_names = mlb.classes_
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        tn = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
        tp = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
        
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
            'support': np.sum(y_true[:, i])
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
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
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
    parser = argparse.ArgumentParser(description='Train multilabel classification model')
    parser.add_argument('--merge-p99-p100', dest='merge_p99_p100', action='store_true', help="Merge classes 'p99' and 'p100' into a single class 'p99_p100'")
    parser.set_defaults(merge_p99_p100=True)
    parser.add_argument('--data-path', type=str, 
                       default="Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--output-path', type=str,
                       default="Data/2_CADICA/models/multilabel",
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
    
    # Create data generators
    train_generator, val_generator, test_generator, mlb = create_data_generators(
        train_df, test_df, val_df,
        batch_size=BATCH_SIZE,
        subsample_fraction=args.subsample
    )
    
    # Create and train model
    logger.info("Creating and training model...")
    model = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS),
        num_classes=len(mlb.classes_)
    )
    
    model, history = train_model(
        model,
        train_generator,
        val_generator,
        num_epochs=args.epochs
    )
    
    # Plot training history
    plot_training_history(history, output_path)
    
    # Evaluate model
    logger.info("\nEvaluating model...")
    evaluate_model(model, test_generator, mlb, output_path)
    
    # Save model and label binarizer
    model_name = f"multilabel_model_subsample_{args.subsample}.h5"
    model.save(output_path / model_name)
    logger.info(f"Model saved to {output_path / model_name}")

if __name__ == "__main__":
    main()

# usage:
# python 2_7_train_multiclass_classification.py  # default settings
# python 2_7_train_multiclass_classification.py --epochs 100  # train for more epochs
# python 2_7_train_multiclass_classification.py --subsample 0.1  # use 10% of data 