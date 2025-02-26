"""
Prepare CADICA dataset for binary classification (lesion/no-lesion videos).
Handles data selection, patient-wise splitting, and reports class distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import logging
from typing import Tuple, Dict
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """Load and filter dataset for binary classification.
    
    Parameters
    ----------
    csv_path : Path
        Path to the frame analysis CSV file
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only selected videos
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Columns: patient_id,video_id,is_lesion,frame_path,frame_id,in_selected_frames,annotations
    
    # Filter for frames selected for each video (so we have the annotations and are validated by a human)
    df_selected = df[df['in_selected_frames'] == True].copy()
    
    
    return df_selected

def analyze_class_distribution(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Analyze class distribution and patient statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool
        Whether to print statistics
        
    Returns
    -------
    Dict
        Dictionary containing distribution statistics
    """
    stats = {
        'total_frames': len(df),
        'total_patients': df['patient_id'].nunique(),
        'total_videos': df['video_id'].nunique(),
        'lesion_frames': df['is_lesion'].sum(),
        'non_lesion_frames': len(df) - df['is_lesion'].sum(),
        'patients_with_lesions': df[df['is_lesion']]['patient_id'].nunique(),
        'patients_without_lesions': df[~df['is_lesion']]['patient_id'].nunique()
    }
    
    if verbose:
        logger.info("\nClass Distribution Statistics:")
        logger.info(f"Total frames: {stats['total_frames']:,}")
        logger.info(f"Total patients: {stats['total_patients']}")
        logger.info(f"Total videos: {stats['total_videos']}")
        logger.info("\nClass Balance:")
        logger.info(f"Lesion frames: {stats['lesion_frames']:,} ({stats['lesion_frames']/stats['total_frames']:.1%})")
        logger.info(f"Non-lesion frames: {stats['non_lesion_frames']:,} ({stats['non_lesion_frames']/stats['total_frames']:.1%})")
        logger.info("\nPatient Distribution:")
        logger.info(f"Patients with lesions: {stats['patients_with_lesions']}")
        logger.info(f"Patients without lesions: {stats['patients_without_lesions']}")
    
    return stats

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
    stats = {
        'train': {
            'total': len(train_df),
            'lesion': train_df['is_lesion'].sum(),
            'non_lesion': len(train_df) - train_df['is_lesion'].sum(),
            'patients': train_df['patient_id'].nunique()
        },
        'test': {
            'total': len(test_df),
            'lesion': test_df['is_lesion'].sum(),
            'non_lesion': len(test_df) - test_df['is_lesion'].sum(),
            'patients': test_df['patient_id'].nunique()
        }
    }
    
    if verbose:
        logger.info("\nTrain/Test Split Statistics:")
        logger.info("\nTraining Set:")
        logger.info(f"Total frames: {stats['train']['total']:,}")
        logger.info(f"Patients: {stats['train']['patients']}")
        logger.info(f"Lesion frames: {stats['train']['lesion']:,} ({stats['train']['lesion']/stats['train']['total']:.1%})")
        logger.info(f"Non-lesion frames: {stats['train']['non_lesion']:,} ({stats['train']['non_lesion']/stats['train']['total']:.1%})")
        
        logger.info("\nTest Set:")
        logger.info(f"Total frames: {stats['test']['total']:,}")
        logger.info(f"Patients: {stats['test']['patients']}")
        logger.info(f"Lesion frames: {stats['test']['lesion']:,} ({stats['test']['lesion']/stats['test']['total']:.1%})")
        logger.info(f"Non-lesion frames: {stats['test']['non_lesion']:,} ({stats['test']['non_lesion']/stats['test']['total']:.1%})")
    
    return stats

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare CADICA dataset for binary classification')
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
    
    # Analyze class distribution
    analyze_class_distribution(df, verbose=not args.quiet)
    
    # Create patient-wise split
    logger.info("\nCreating patient-wise train/test split...")
    train_df, test_df = create_patient_wise_split(
        df, 
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Analyze split distribution
    analyze_split_distribution(train_df, test_df, verbose=not args.quiet)
    
    # Return DataFrames for further use
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = main()
    
# usage:
# python 2_prepare_binary_classification.py  # default settings
# python 2_prepare_binary_classification.py --quiet  # suppress detailed stats
# python 2_prepare_binary_classification.py --test-size 0.3  # adjust split ratio
# python 2_prepare_binary_classification.py --seed 123  # change random seed 