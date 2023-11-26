import os
import requests
import tarfile
import re




# Define function to download file from a URL
def download_file(url, filename):
	with requests.get(url, stream=True) as response:
		response.raise_for_status()
		with open(filename, 'wb') as file:
			for chunk in response.iter_content(chunk_size=8192):
				file.write(chunk)


# Define function to extract tar.gz files
def extract_tar_gz(filename, extract_folder):
	with tarfile.open(filename, 'r:gz') as tar:
		tar.extractall(path=extract_folder)

markers = ['golgi', 'mito', 'actin', 'lysosome', 'tublin']

for marker in markers:
	BASE_DIR = f"{os.getcwd()}/data/AICS/AICS_{marker}"

	# Ensure the base directory exists
	if not os.path.exists(BASE_DIR):
		os.makedirs(BASE_DIR)
	
	# Read URLs from the file
	with open(f'{os.getcwd()}/data/AICS/AICS_{marker}/aics_img_url_list.txt', 'r') as file:
		urls = [line.strip() for line in file]
	
	# Process each URL
	for url in urls:
		# Extract ID from the URL using regex
		match = re.search(r'id=F(\d+)', url)
		if not match:
			print(f"Could not extract ID from URL: {url}")
			continue
		
		id = match.group(1)
		filename = os.path.join(BASE_DIR, f"AICS_{id}.tar.gz")
		extract_folder = os.path.join(BASE_DIR, f"AICS_{id}")
		
		# Download file
		print(f"Downloading {url} to {filename}...")
		download_file(url, filename)
		
		# Extract tar.gz file
		print(f"Extracting {filename} to {extract_folder}...")
		extract_tar_gz(filename, extract_folder)
		
		# Optionally, remove the downloaded tar.gz file after extraction
		os.remove(filename)
	
	print("All files processed.")
