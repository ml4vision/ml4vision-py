import requests
import os
from urllib.request import urlretrieve
import json
from multiprocessing import Pool
from tqdm import tqdm
from itertools import repeat   
from ml4vision.utils import mask_utils 

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

    def load_label_and_prediction(self):
        sample_details = self.client.get(f'/samples/{self.uuid}/')
        self.label = sample_details['label']
        self.prediction = sample_details['prediction']

    def pull(self, location='./', format='json'):
        asset_filename = self.asset['filename']
        asset_location = os.path.join(location, 'images', asset_filename)
        if not os.path.exists(asset_location):
            urlretrieve(self.asset['url'], asset_location)

        self.load_label_and_prediction()

        if format == "mask":
            label_filename = os.path.splitext(asset_filename)[0] + '.png'
            label_location = os.path.join(location, 'labels', label_filename)
            size = self.metadata['size']
            
            if self.label or self.prediction:
                mask = mask_utils.annotations_to_mask(self.label['annotations'] or self.prediction['annotations'], size)
            else:
                mask = mask_utils.empty_mask(size)

            mask.save(label_location)

        else: # json
            label_filename = os.path.splitext(asset_filename)[0] + '.json'
            label_location = os.path.join(location, 'labels', label_filename)

            if self.label or self.prediction:
                with open(label_location, 'w') as f:
                    json.dump(self.label or self.prediction, f)



    def delete(self):
        self.client.delete(f'/samples/{self.uuid}/')

class Project:
    
    def __init__(self, client, **project_data):
        self.client = client
        for key, value in project_data.items():
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

    def push(self, image_list):
        
        print('Uploading data')
        with Pool(8) as p:
            inputs = zip(repeat(self), image_list)
            r = p.starmap(Project.create_sample, tqdm(inputs, total=len(image_list)))


    def list_samples(self, filter=None):
        samples = []
        
        page=1
        while(True):
            try:
                endpoint = f'/projects/{self.uuid}/samples/?page={page}'
                if filter:
                    endpoint += ('&' + filter)
                for sample in self.client.get(endpoint):
                    samples.append(Sample(self.client, **sample))
                page+=1
            except:
                break
        
        return samples

    def create_sample(self, image_file):
        # create asset
        filename = os.path.basename(image_file)
        payload = {
            'filename': filename
        }
        asset_data = self.client.post(f'/assets/', payload=payload)

        # upload file to s3
        url = asset_data['presigned_post_fields']['url']
        fields = asset_data['presigned_post_fields']['fields']
        
        with open(image_file, 'rb') as f:
            response = requests.post(url, data=fields, files={'file':f})
        
        if response.status_code != 204:
            raise Exception(f"Failed uploading to s3, status_code: {response.status_code}")

        # create sample
        payload = {
            'name': filename,
            'asset': asset_data['uuid']
        }
        sample_data = self.client.post(f'/projects/{self.uuid}/samples/', payload)

        return Sample(client=self.client, **sample_data)

    def delete(self):
        self.client.delete(f'/projects/{self.uuid}/')

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

    def list_projects(self):
        projects = []
        
        page=1
        while(True):
            try:
                for project_data in self.get(f'/projects/?page={page}'):
                    projects.append(Project(self, **project_data))
                page+=1
            except:
                break
        
        return projects

    def get_project_by_uuid(self, project_uuid):
        project_data = self.get(f'/projects/{project_uuid}/')
        return Project(self, **project_data)

    def get_project_by_name(self, name, owner=None):
        owner = owner if owner else self.username
        project_data = self.get(f'/projects/?name={name}&owner={owner}')
        
        if len(project_data) == 0:
            raise Exception(f'Did not found project "{name}" for owner "{owner}". If this is a shared project, please specify the owner')

        return Project(self, **project_data[0])

    def create_project(self, name, description='', categories=[{'id': 0, 'name': 'object'}] ,annotation_type='BBOX', model=''):
        payload = {
            'name': name,
            'description': description,
        }
        if categories:
            payload['categories'] = categories
        if annotation_type:
            payload['annotation_type'] = annotation_type
        if model:
            payload['model'] = model
        
        project_data = self.post('/projects/', payload)
        
        return Project(self, **project_data)
