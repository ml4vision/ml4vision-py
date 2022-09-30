import os
from pathlib import Path
from ml4vision.client import Client
import glob
import sys

def _save_apikey(apikey):
    config_path = Path().home() / '.ml4vision' / 'credentials'
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(f"apikey={apikey}\n")

def _load_apikey():
    config_path = Path().home() / '.ml4vision' / 'credentials'
    
    try:
        with open(config_path, 'r') as f:
            apikey = f.readline().split('=',1)[1].strip()
            return apikey
    except:
        print('No API Key detected. First authenticate.')
        sys.exit(1)

def _get_client():
    apikey = _load_apikey()
    client = Client(apikey)
    return client

def authenticate(apikey):
    try:
        client = Client(apikey)
        _save_apikey(apikey)
    except:
        print('API Key is invalid.')
        sys.exit(1)

def pull_project(name, approved_only, labeled_only):
    client = _get_client() 
    try:   
        project = client.get_project_by_name(name)
        project.load_samples(labeled_only=labeled_only, approved_only=approved_only)
        project.pull()
    except Exception as e:
        print(e)
        sys.exit(1)

def push_to_project(name, path, label_path=None):
    client = _get_client()
    try:
        project = client.get_project_by_name(name)
        
        image_files = []
        image_types = (
            '*.png', '*.PNG', 
            '*.jpg','*.JPG', 
            '*.jpeg', '*.JPEG', 
            '*.tiff', '*.TIFF', 
            '*.tif', '*.TIF'
        )
        for t in image_types:
            image_files.extend(glob.glob(os.path.join(path, t)))
        image_files.sort()

        if label_path:
            label_files = []
            label_types = (
                '*.png', '*.PNG',
                '*tif', '*.TIF'
            )
            for t in label_types:
                label_files.extend(glob.glob(os.path.join(label_path, t)))
            label_files.sort()

            assert len(image_files) == len(label_files), f'Number of images ({len(image_files)}) does not equal number of labels ({len(label_files)})'
            project.push(image_files, label_files)
        else:
            project.push(image_files)


    except Exception as e:
        print(e)
        sys.exit(1)

def list_projects():
    client = _get_client()
    projects = client.list_projects()
    for dts in projects:
        print(dts.name)