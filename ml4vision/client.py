import requests
import os
from urllib.request import urlretrieve
import json
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat   
from ml4vision.utils import mask_utils 
from PIL import Image
import numpy as np

class Sample:

    def __init__(self, client, **kwargs):
        self.client  = client
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_label(self, label):
        payload = {
            'annotations': label
        }
        label = self.client.put(f'/samples/{self.uuid}/label/', payload=payload)
        self.label = label

    def load_label(self):
        sample_details = self.client.get(f'/samples/{self.uuid}/')
        self.label = sample_details['label']

    def pull(self, location='./', format='json'):
        asset_filename = self.asset['filename']
        asset_location = os.path.join(location, 'images', asset_filename)
        if not os.path.exists(asset_location):
            urlretrieve(self.asset['url'], asset_location)

        if self.label is not None:
            self.load_label()

            if format == "mask":
                label_filename = os.path.splitext(asset_filename)[0] + '.png'
                label_location = os.path.join(location, 'labels', label_filename)
                size = self.asset['metadata']['size']
                
                mask = mask_utils.annotations_to_label(self.label['annotations'], size)
                mask.save(label_location)

            else: # json
                label_filename = os.path.splitext(asset_filename)[0] + '.json'
                label_location = os.path.join(location, 'labels', label_filename)

                with open(label_location, 'w') as f:
                    json.dump(self.label, f)

    def delete(self):
        self.client.delete(f'/samples/{self.uuid}/')

class Dataset:
    
    def __init__(self, client, **dataset_data):
        self.client = client
        for key, value in dataset_data.items():
            setattr(self, key, value)

    def pull(self, location='./', format="json", approved_only=False):

        dataset_loc = os.path.join(location, self.name)
        image_loc = os.path.join(dataset_loc, 'images')
        label_loc = os.path.join(dataset_loc, 'labels')

        os.makedirs(image_loc, exist_ok=True)
        os.makedirs(label_loc, exist_ok=True)

        # download all samples & labels
        print('Gathering all samples...')
        if approved_only:
            samples = self.list_samples(filter='approved=True')
        else:
            samples = self.list_samples()

        print('Downloading your dataset...')
        with Pool(8) as p:
            inputs = zip(samples, repeat(dataset_loc), repeat(format))
            r = p.starmap(Sample.pull, tqdm(inputs, total=len(samples)))

    def push(self, image_list, label_list=None):
        
        print('Uploading data')
        with Pool(8) as p:
            if label_list:
                inputs = zip(repeat(self), image_list, label_list)
            else:
                inputs = zip(repeat(self), image_list)
            r = p.starmap(Dataset.create_sample, tqdm(inputs, total=len(image_list)))

    def list_samples(self, filter=None):
        samples = []
        
        page=1
        while(True):
            try:
                endpoint = f'/datasets/{self.uuid}/samples/?page={page}'
                if filter:
                    endpoint += ('&' + filter)
                for sample in self.client.get(endpoint):
                    samples.append(Sample(self.client, **sample))
                page+=1
            except:
                break
        
        return samples

    def create_sample(self, image_file, label_file=None):
        # create asset
        filename = os.path.basename(image_file)
        payload = {
            'filename': filename,
            'type': 'IMAGE'
        }
        asset_data = self.client.post(f'/assets/', payload=payload)

        # upload file to s3
        url = asset_data['presigned_post_fields']['url']
        fields = asset_data['presigned_post_fields']['fields']
        
        with open(image_file, 'rb') as f:
            response = requests.post(url, data=fields, files={'file':f})
        
        if response.status_code != 204:
            raise Exception(f"Failed uploading to s3, status_code: {response.status_code}")

        # confirm upload
        self.client.put(f'/assets/{asset_data["uuid"]}/confirm_upload/')

        # create sample
        payload = {
            'name': filename,
            'asset': asset_data['uuid']
        }
        sample_data = self.client.post(f'/datasets/{self.uuid}/samples/', payload)

        sample = Sample(client=self.client, **sample_data)

        # upload label
        if label_file:
            label = np.array(Image.open(label_file))
            annotations = mask_utils.label_to_annotations(label)
            sample.update_label(annotations)

        return sample

    def delete(self):
        self.client.delete(f'/datasets/{self.uuid}/')

class Client:

    def __init__(self, apikey, url="https://api.ml4vision.com"):

        self.url = url
        self.apikey = apikey
        self.username, self.email = self.get_owner()

    def get_owner(self):
        owner = self.get('/auth/users/me')
        return owner['username'], owner['email']

    def get(self, endpoint):

        response = requests.get(self.url + endpoint, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code}")

    def post(self, endpoint, payload={}, files=None):

        response = requests.post(self.url + endpoint, json=payload, files=files, headers=self._get_headers())

        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code} - {response.text}")

    def put(self, endpoint, payload={}):

        response = requests.put(self.url + endpoint, json=payload, headers = self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed, status_code: {response.status_code} - {response.text}")


    def delete(self, endpoint):
        response = requests.delete(self.url + endpoint, headers=self._get_headers())
        
        if response.status_code != 204:
            raise Exception(f"Request failed, status_code: {response.status_code}")

    def _get_headers(self):

        # set content type & authorization token
        headers = {
            'Authorization': f"APIKey {self.apikey}"
        }

        return headers

    def list_datasets(self):
        datasets = []
        
        page=1
        while(True):
            try:
                for dataset_data in self.get(f'/datasets/?page={page}'):
                    datasets.append(Dataset(self, **dataset_data))
                page+=1
            except:
                break
        
        return datasets

    def get_dataset_by_uuid(self, dataset_uuid):
        dataset_data = self.get(f'/datasets/{dataset_uuid}/')
        return Dataset(self, **dataset_data)

    def get_dataset_by_name(self, name, owner=None):
        owner = owner if owner else self.username
        dataset_data = self.get(f'/datasets/?name={name}&owner={owner}')
        
        if len(dataset_data) == 0:
            raise Exception(f'Did not found dataset "{name}" for owner "{owner}". If this is a shared or public dataset, please specify the owner')

        return Dataset(self, **dataset_data[0])

    def create_dataset(self, name, description='', categories=[{'id': 0, 'name': 'object', 'has_instances': True}] ,annotation_type='BBOX'):
        payload = {
            'name': name,
            'description': description,
        }
        if categories:
            payload['categories'] = categories
        if annotation_type:
            payload['annotation_type'] = annotation_type

        dataset_data = self.post('/datasets/', payload)
        
        return Dataset(self, **dataset_data)
