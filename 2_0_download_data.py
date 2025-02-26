"""
Download and extract dataset from specified URL.

This script downloads a large dataset from an S3 bucket and saves it to a local directory.
It includes progress tracking, error handling, and verification of the download.
"""

import os
import requests
import logging
from pathlib import Path
from tqdm import tqdm
import zipfile
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_directory(path: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str
        Path to create
        
    Returns
    -------
    Path
        Path object of created directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def file_exists_and_valid(file_path: Path, expected_size: int = None) -> bool:
    """
    Check if file exists and has the expected size.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to check
    expected_size : int, optional
        Expected file size in bytes
        
    Returns
    -------
    bool
        True if file exists and is valid
    """
    if not file_path.exists():
        return False
    
    if expected_size and file_path.stat().st_size != expected_size:
        logger.warning(f"Existing file size mismatch. Expected: {expected_size}, Found: {file_path.stat().st_size}")
        return False
        
    return True

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download file from URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    destination : Path
        Path to save file to
    chunk_size : int, optional
        Size of chunks to download, by default 8192
    
    Raises
    ------
    Exception
        If download fails or connection errors occur
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if file already exists and has correct size
        if file_exists_and_valid(destination, total_size):
            logger.info(f"File already exists and is valid: {destination}")
            return
            
        with open(destination, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {destination.name}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
                    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {str(e)}")

def extract_zip(zip_path: Path, extract_path: Path) -> None:
    """
    Extract zip file to specified path.
    
    Parameters
    ----------
    zip_path : Path
        Path to zip file
    extract_path : Path
        Path to extract to
        
    Raises
    ------
    Exception
        If extraction fails
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            
            # Get the main content file (CADICA.zip)
            main_zip = next((f for f in files if f.endswith('CADICA.zip')), None)
            if not main_zip:
                raise Exception("CADICA.zip not found in the downloaded archive")
            
            # Extract only the CADICA.zip file
            logger.info("Extracting main CADICA.zip file...")
            zip_ref.extract(main_zip, extract_path)
            
            # Move CADICA.zip to the correct location
            nested_path = extract_path / main_zip
            final_path = extract_path / "CADICA.zip"
            nested_path.rename(final_path)
            
            # Remove the extra directory structure
            extra_dir = extract_path / "Data"
            if extra_dir.exists():
                import shutil
                shutil.rmtree(extra_dir)
            
            logger.info(f"Successfully moved CADICA.zip to {final_path}")
            
            # Now extract the contents of CADICA.zip
            logger.info("Extracting contents of CADICA.zip...")
            with zipfile.ZipFile(final_path, 'r') as inner_zip:
                inner_files = inner_zip.namelist()
                with tqdm(total=len(inner_files), desc="Extracting dataset files") as pbar:
                    for file in inner_files:
                        inner_zip.extract(file, extract_path)
                        pbar.update(1)
                        
    except zipfile.BadZipFile:
        raise Exception("Invalid or corrupted zip file")
    except Exception as e:
        raise Exception(f"Extraction failed: {str(e)}")

def main():
    """Main function to handle download and extraction process."""
    
    # Configuration
    url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/p9bpx9ctcv-2.zip"
    data_dir = create_directory("Data/2_CADICA")
    zip_path = data_dir / "downloaded_archive.zip"  # Temporary name for downloaded zip
    
    logger.info("Starting download process...")
    
    try:
        # Download the file
        download_file(url, zip_path)
        
        logger.info("Download completed. Starting extraction...")
        
        # Extract the file
        extract_zip(zip_path, data_dir)
        
        logger.info("Extraction completed successfully!")
        
        # Clean up the downloaded archive
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Downloaded archive removed.")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return
    
if __name__ == "__main__":
    main()
