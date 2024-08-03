"""
This script is for downloading segmentation masks and evaluation metrics
from Zenodo. It is a necessary step if one wants to directly plot figures.
"""

import requests
import os
import zipfile

# URLs of the files to be downloaded
file_urls = [
    "https://zenodo.org/record/12859553/files/evaluation_metrics.zip?download=1"
    "https://zenodo.org/record/12859553/files/segmentation_masks.zip?download=1"
]

# Download each file to the ../data directory and unzip
for url in file_urls:
    local_filename = os.path.join('../data', url.split('/')[-1].split('?')[0])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"{local_filename} downloaded")
    
    # Unzip the file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall('../data')
    print(f"{local_filename} unzipped")

print("All files downloaded and unzipped successfully.")
