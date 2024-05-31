import time
import os
import requests, io, zipfile
from distutils.dir_util import copy_tree

def download_data():
    start_time = time.time()
    
    # Prepare data
    os.makedirs('data/raw/images_1', exist_ok=True)
    os.makedirs('data/raw/images_2', exist_ok=True)
    os.makedirs('data/raw/images_all', exist_ok=True)
    
    metadata_path = 'data/raw/metadata.csv'
    image_path_1 = 'data/raw/images_1.zip'
    image_path_2 = 'data/raw/images_2.zip'
    images_rgb_path = 'data/raw/hmnist_8_8_RGB.csv'
    
    urls = {
        'metadata.csv': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv',
        'images_1.zip': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip',
        'images_2.zip': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip',
        'hmnist_8_8_RGB.csv': 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
    }
    
    # Download files
    for file_name, url in urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'data/raw/{file_name}', 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {file_name} from {url}")
    
    # Unzip image files
    os.system('unzip -q -o data/raw/images_1.zip -d data/raw/images_1')
    os.system('unzip -q -o data/raw/images_2.zip -d data/raw/images_2')
    
    # Merge images into one directory
    copy_tree('data/raw/images_1', 'data/raw/images_all')
    copy_tree('data/raw/images_2', 'data/raw/images_all')
    
    print("Downloaded and prepared data.")
    print("Execution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    download_data()
