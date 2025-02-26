"""
Prepare CADICA dataset for multiclass classification using annotation labels.
Handles data selection, patient-wise splitting, and reports class distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import logging
from typing import Tuple, Dict, List
import argparse
import ast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_annotations(annotations_str: str) -> List[Dict]:
    """Parse annotations string into list of dictionaries.
    
    Parameters
    ----------
    annotations_str : str
        String representation of annotations list
        
    Returns
    -------
    List[Dict]
        List of annotation dictionaries
    """
    try:
        if pd.isna(annotations_str) or annotations_str == '[]':
            return []
        return ast.literal_eval(annotations_str)
    except:
        logger.warning(f"Failed to parse annotations: {annotations_str}")
        return []

def extract_unique_classes(df: pd.DataFrame) -> List[str]:
    """Extract unique class labels from annotations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing annotations column
        
    Returns
    -------
    List[str]
        List of unique class labels
    """
    unique_classes = set()
    
    for annotations_str in df['annotations']:
        annotations = parse_annotations(annotations_str)
        for ann in annotations:
            if 'class' in ann:
                unique_classes.add(ann['class'])
    
    return sorted(list(unique_classes))

def analyze_annotation_distribution(df: pd.DataFrame) -> None:
    """Analyze and print distribution of annotation classes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing annotations
    """
    # Parse all annotations
    class_counts = {}
    empty_annotations = 0
    total_frames = len(df)
    
    for annotations_str in df['annotations']:
        annotations = parse_annotations(annotations_str)
        if not annotations:
            empty_annotations += 1
            continue
            
        for ann in annotations:
            if 'class' in ann:
                class_name = ann['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info("\nAnnotation Distribution Analysis:")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Frames without annotations: {empty_annotations} ({empty_annotations/total_frames:.1%})")
    logger.info(f"Frames with annotations: {total_frames - empty_annotations} ({(total_frames - empty_annotations)/total_frames:.1%})")
    logger.info("\nClass Distribution:")
    
    for class_name, count in sorted(class_counts.items()):
        logger.info(f"{class_name}: {count} ({count/total_frames:.1%})")

def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """Load and filter dataset for multiclass classification.
    
    Parameters
    ----------
    csv_path : Path
        Path to the frame analysis CSV file
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only frames with valid annotations
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter for frames selected for each video
    df_selected = df[df['in_selected_frames'] == True].copy()
    
    # Parse annotations and filter out empty ones
    df_selected['parsed_annotations'] = df_selected['annotations'].apply(parse_annotations)
    df_selected = df_selected[df_selected['parsed_annotations'].apply(len) > 0]
    
    logger.info(f"Total frames: {len(df)}")
    logger.info(f"Selected frames: {len(df_selected)}")
    logger.info(f"Frames with annotations: {len(df_selected)}")
    
    # Analyze annotation distribution
    analyze_annotation_distribution(df_selected)
    
    return df_selected

def create_patient_wise_split(
    df: pd.DataFrame, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create patient-wise train/test split to avoid data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    test_size : float
        Proportion of patients to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames
    """
    # Initialize the splitter
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Split based on patient_id
    train_idx, test_idx = next(splitter.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Verify no patient overlap
    train_patients = set(train_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    assert len(train_patients.intersection(test_patients)) == 0, "Patient overlap detected!"
    
    return train_df, test_df

def analyze_split_distribution(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """Analyze class distribution in train/test splits.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Testing DataFrame
    verbose : bool
        Whether to print statistics
        
    Returns
    -------
    Dict
        Split statistics
    """
    def get_class_distribution(df):
        class_counts = {}
        for annotations in df['parsed_annotations']:
            for ann in annotations:
                if 'class' in ann:
                    class_name = ann['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts
    
    train_dist = get_class_distribution(train_df)
    test_dist = get_class_distribution(test_df)
    
    if verbose:
        logger.info("\nTrain/Test Split Statistics:")
        logger.info(f"\nTraining Set (Total: {len(train_df)} frames):")
        for class_name, count in sorted(train_dist.items()):
            logger.info(f"{class_name}: {count} ({count/len(train_df):.1%})")
        
        logger.info(f"\nTest Set (Total: {len(test_df)} frames):")
        for class_name, count in sorted(test_dist.items()):
            logger.info(f"{class_name}: {count} ({count/len(test_df):.1%})")
    
    return {
        'train': train_dist,
        'test': test_dist
    }

def main():
    parser = argparse.ArgumentParser(description='Prepare CADICA dataset for multiclass classification')
    parser.add_argument('--path', type=str, 
                       default="Data/2_CADICA/cadica_frame_analysis.csv",
                       help='Path to frame analysis CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of patients for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed statistics')
    args = parser.parse_args()
    
    # Load and filter data
    logger.info("Loading and filtering dataset...")
    df = load_and_filter_data(Path(args.path))
    
    # Create patient-wise split
    logger.info("\nCreating patient-wise train/test split...")
    train_df, test_df = create_patient_wise_split(
        df, 
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Analyze split distribution
    analyze_split_distribution(train_df, test_df, verbose=not args.quiet)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = main()

# usage:
# python 2_6_prepare_multiclass_classification.py  # default settings
# python 2_6_prepare_multiclass_classification.py --quiet  # suppress detailed stats
# python 2_6_prepare_multiclass_classification.py --test-size 0.3  # adjust split ratio 