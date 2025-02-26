"""
Create DataFrame from CADICA dataset by processing selected videos, frames, and annotations.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_lesion_status(patient_dir: Path) -> Dict[str, Set[str]]:
    """Get lesion/non-lesion video IDs for a patient directory.
    
    Parameters
    ----------
    patient_dir : Path
        Path to patient directory (pX)
        
    Returns
    -------
    Dict[str, Set[str]]
        Dictionary with 'lesion' and 'nonlesion' video ID sets
    """
    status = {'lesion': set(), 'nonlesion': set()}
    
    try:
        with open(patient_dir / 'lesionVideos.txt') as f:
            status['lesion'] = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"Missing lesionVideos.txt in {patient_dir}")
    
    try:
        with open(patient_dir / 'nonlesionVideos.txt') as f:
            status['nonlesion'] = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"Missing nonlesionVideos.txt in {patient_dir}")
    
    return status

def parse_annotation_line(line: str) -> Optional[Dict]:
    """Parse annotation line containing bounding box and class.
    
    Parameters
    ----------
    line : str
        Raw annotation line (e.g., "257 123 42 55 p0_20")
        
    Returns
    -------
    Optional[Dict]
        Parsed annotation with bbox and class
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    
    try:
        return {
            'bbox': [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])],
            'class': parts[4]
        }
    except (ValueError, IndexError):
        logger.warning(f"Invalid annotation format: {line}")
        return None

def load_groundtruth_annotations(video_dir: Path) -> Dict[str, List[Dict]]:
    """Load ground truth annotations for video frames.
    
    Parameters
    ----------
    video_dir : Path
        Path to video directory
        
    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping frame IDs to their annotations
    """
    annotations = {}
    gt_dir = video_dir / 'groundtruth'
    
    if not gt_dir.exists():
        return annotations
    
    for ann_file in gt_dir.glob('*.txt'):
        try:
            frame_id = ann_file.stem.split('_')[-1]
            with open(ann_file) as f:
                frame_anns = []
                for line in f:
                    ann = parse_annotation_line(line.strip())
                    if ann:
                        frame_anns.append(ann)
                if frame_anns:
                    annotations[frame_id] = frame_anns
        except Exception as e:
            logger.error(f"Error processing {ann_file}: {str(e)}")
    
    return annotations

def get_selected_frames(video_dir: Path, patient_id: str, video_id: str) -> Set[str]:
    """Get selected frame IDs from video directory.
    
    Parameters
    ----------
    video_dir : Path
        Path to video directory
    patient_id : str
        Patient ID (pX)
    video_id : str
        Video ID (vY)
        
    Returns
    -------
    Set[str]
        Set of selected frame IDs
    """
    selected_file = video_dir / f"{patient_id}_{video_id}_selectedFrames.txt"
    if not selected_file.exists():
        logger.warning(f"Missing selected frames file: {selected_file}")
        return set()
    
    try:
        with open(selected_file) as f:
            return {line.strip().split('_')[-1] for line in f if line.strip()}
    except Exception as e:
        logger.error(f"Error reading {selected_file}: {str(e)}")
        return set()

def process_video_directory(video_dir: Path, patient_id: str, video_id: str, 
                          is_lesion: bool) -> List[Dict]:
    """Process video directory to extract frame data.
    
    Parameters
    ----------
    video_dir : Path
        Path to video directory
    patient_id : str
        Patient ID
    video_id : str
        Video ID
    is_lesion : bool
        Whether video contains lesions
        
    Returns
    -------
    List[Dict]
        List of frame records with annotations
    """
    records = []
    
    # Get selected frames and annotations
    selected_frames = get_selected_frames(video_dir, patient_id, video_id)
    annotations = load_groundtruth_annotations(video_dir)
    
    # Process input frames
    input_dir = video_dir / 'input'
    if not input_dir.exists():
        logger.warning(f"Missing input directory: {input_dir}")
        return records
    
    for frame_file in input_dir.glob('*.png'):
        try:
            frame_id = frame_file.stem.split('_')[-1]
            record = {
                'patient_id': patient_id,
                'video_id': video_id,
                'is_lesion': is_lesion,
                'frame_path': str(frame_file.resolve()),
                'frame_id': frame_id,
                'in_selected_frames': frame_id in selected_frames,
                'annotations': annotations.get(frame_id, [])
            }
            records.append(record)
        except Exception as e:
            logger.error(f"Error processing {frame_file}: {str(e)}")
    
    return records

def create_dataset_dataframe(base_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Create DataFrame from CADICA dataset structure.
    
    Parameters
    ----------
    base_path : Path
        Root path of CADICA dataset
    verbose : bool, optional
        Whether to print processing logs, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing frame-level data with annotations
    """
    records = []
    
    # Process selected videos
    selected_path = base_path / 'selectedVideos'
    if not selected_path.exists():
        logger.error(f"Selected videos directory not found: {selected_path}")
        return pd.DataFrame()
    
    # Process each patient directory
    for patient_dir in selected_path.glob('p*'):
        if not patient_dir.is_dir():
            continue
            
        patient_id = patient_dir.name
        if verbose:
            logger.info(f"Processing patient {patient_id}...")
        
        # Get lesion status for videos
        lesion_status = get_lesion_status(patient_dir)
        
        # Process each video directory
        for video_dir in patient_dir.glob('v*'):
            if not video_dir.is_dir():
                continue
                
            video_id = video_dir.name
            is_lesion = video_id in lesion_status['lesion']
            
            if verbose:
                logger.info(f"Processing {patient_id}/{video_id} (lesion: {is_lesion})")
            video_records = process_video_directory(
                video_dir, patient_id, video_id, is_lesion
            )
            records.extend(video_records)
    
    return pd.DataFrame(records)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create CADICA dataset DataFrame')
    parser.add_argument('--quiet', action='store_true', help='Suppress processing logs')
    parser.add_argument('--path', type=str, default="Data/2_CADICA/CADICA",
                       help='Path to CADICA dataset')
    args = parser.parse_args()
    
    # Set dataset root path
    base_path = Path(args.path)
    
    # Create DataFrame
    logger.info("Creating dataset DataFrame...")
    df = create_dataset_dataframe(base_path, verbose=not args.quiet)
    
    # Save to CSV in the parent directory of the dataset
    csv_path = base_path.parent / "cadica_frame_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"DataFrame saved to {csv_path}")
    
    # Print summary
    logger.info(f"\nDataset Summary:")
    logger.info(f"Total patients: {df['patient_id'].nunique()}")
    logger.info(f"Total videos: {df['video_id'].nunique()}")
    logger.info(f"Total frames: {len(df)}")
    logger.info(f"Frames with annotations: {df[df['annotations'].apply(len) > 0].shape[0]}")

if __name__ == "__main__":
    main() 
    
# usage: 
# 
# verbose output by default:
# python 2_1_create_dataframe.py
# 
# quiet output:
# python 2_1_create_dataframe.py --quiet
# 
# specify path:
# python 2_1_create_dataframe.py --path "Data/2_CADICA/CADICA"
# 
# specify path and quiet output:
# python 2_1_create_dataframe.py --path "Data/2_CADICA/CADICA" --quiet
