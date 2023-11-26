import requests
import os

def download_file(url, destination):
	# Ensure the target directory exists
	os.makedirs(os.path.dirname(destination), exist_ok=True)

	# Send a GET request to the URL
	response = requests.get(url, stream=True)

	# Check if the request was successful
	if response.status_code == 200:
		with open(destination, 'wb') as file:
			for chunk in response.iter_content(chunk_size=1024):
				file.write(chunk)
		print(f"File downloaded successfully and saved as {destination}")
	else:
		print(f"Failed to download file: status code {response.status_code}")

# URL of the file to download
hubmap_IDs = ['a296c763352828159f3adfa495becf3e', 'cd880c54e0095bad5200397588eccf81', 'd3130f4a89946cc6b300b115a3120b7a']
for hubmap_ID in hubmap_IDs:
	
	url = f"https://g-d00e7b.09193a.5898.dn.glob.us/{hubmap_ID}/data/3D_image_stack.ome.tiff?download=1"
	
	# Destination file path
	destination = os.path.join(os.getcwd(), "data", "IMC_3D", "florida-3d-imc", hubmap_ID, "3D_image_stack.ome.tiff")
	
	# Download the file
	download_file(url, destination)
