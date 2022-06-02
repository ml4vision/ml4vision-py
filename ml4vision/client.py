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

class MLModel:

    def __init__(self, client, **kwargs):
        self.client = client
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_version(self, model_file, params={}):

        # create asset
        filename = os.path.basename(model_file)
        payload = {
            'filename': filename,
        }
        asset_data = self.client.post(f'/assets/', payload=payload)

        # upload file to s3
        url = asset_data['presigned_post_fields']['url']
        fields = asset_data['presigned_post_fields']['fields']
        
        with open(model_file, 'rb') as f:
            response = requests.post(url, data=fields, files={'file':f})
        
        if response.status_code != 204:
            raise Exception(f"Failed uploading to s3, status_code: {response.status_code}")

        # confirm upload
        self.client.put(f'/assets/{asset_data["uuid"]}/confirm_upload/')

        # create version
        payload = {
            'asset': asset_data['uuid'],
            'params': params
        }
        version_data = self.client.post(f'/models/{self.uuid}/versions/', payload)

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
        if self.label is not None:
            sample_details = self.client.get(f'/samples/{self.uuid}/')
            self.label = sample_details['label']
        # return object for multiprocessing
        return self

    def pull_image(self, location='./'):
        asset_filename = self.asset['filename']
        asset_location = os.path.join(location, 'images', asset_filename)
        if not os.path.exists(asset_location):
            urlretrieve(self.asset['url'], asset_location)

    def pull_label(self, location='./', as_json=True, type='BBOX'):
        self.load_label()

        if as_json or type=='BBOX':

            asset_filename = self.asset['filename']
            label_filename = os.path.splitext(asset_filename)[0] + '.json'
            label_location = os.path.join(location, 'labels', label_filename)

            with open(label_location, 'w') as f:
                json.dump(self.label, f)

        elif type == "SEGMENTATION":
            asset_filename = self.asset['filename']
            label_filename = os.path.splitext(asset_filename)[0] + '.png'
            label_location = os.path.join(location, 'labels', label_filename)
            size = self.asset['metadata']['size']

            _, cls = mask_utils.annotations_to_label(self.label['annotations'], size)
            cls.save(label_location)
            
        elif type == "INSTANCE_SEGMENTATION":

            asset_filename = self.asset['filename']
            inst_filename = os.path.splitext(asset_filename)[0] + '_inst.png'
            inst_location = os.path.join(location, 'labels', inst_filename)
            cls_filename = os.path.splitext(asset_filename)[0] + '_cls.png'
            cls_location = os.path.join(location, 'labels', cls_filename)
            size = self.asset['metadata']['size']

            inst, cls = mask_utils.annotations_to_label(self.label['annotations'], size)
            inst.save(inst_location)
            cls.save(cls_location)

        else:
            assert False, f'type {type} is not implemented!'

    def pull(self, location='./', as_json=True, type='BBOX'):
        self.pull_image(location=location)
        self.pull_label(location=location, as_json=as_json, type=type)

    def delete(self):
        self.client.delete(f'/samples/{self.uuid}/')

class Dataset:
    
    def __init__(self, client, **dataset_data):
        self.client = client
        for key, value in dataset_data.items():
            setattr(self, key, value)

    def pull(self, location='./', as_json=False, images_only=False, labels_only=False):
        dataset_loc = os.path.join(location, self.name)

        if format == 'coco':
            image_loc = os.path.join(dataset_loc, 'images')
            os.makedirs(image_loc, exist_ok=True)
            
            print('Loading images')
            with Pool(8) as p:
                inputs = zip(self.samples, repeat(dataset_loc))
                r = p.starmap(Sample.pull_image, tqdm(inputs, total=len(self.samples)))

            print('Loading labels')
            with Pool(8) as p:
                r = p.map(Sample.load_label, tqdm(self.samples, total=len(self.samples)))
                self.samples = r
            
            print('Creating coco json file')
            images = []
            annotations = []

            for sample in self.samples:
                image = {
                    'id': sample.name.rsplit('.', 1)[0],
                    'file_name': sample.name,
                    'width': sample.asset['metadata']['size'][0],
                    'height': sample.asset['metadata']['size'][1]
                }
                images.append(image)

                if sample.label is not None:
                    for i, item in enumerate(sample.label['annotations']):
                        ann = {
                            'id': f"{sample.name.rsplit('.', 1)[0]}_{i}",
                            'image_id': sample.name.rsplit('.', 1)[0],
                            'bbox': item['bbox'],
                            'area': item['area'],
                            'category_id': item['category_id']
                        }
                        annotations.append(ann)

            with open(f'{os.path.join(dataset_loc, self.name)}.json', 'w') as f:
                coco = {
                    'images': images,
                    'categories': self.categories,
                    'annotations': annotations
                }
                json.dump(coco, f)

        else:
            if images_only:
                image_loc = os.path.join(dataset_loc, 'images')
                os.makedirs(image_loc, exist_ok=True)

                print('Downloading your dataset...')
                with Pool(8) as p:
                    inputs = zip(self.samples, repeat(dataset_loc))
                    r = p.starmap(Sample.pull_image, tqdm(inputs, total=len(self.samples)))

                return dataset_loc
            
            elif labels_only:
                label_loc = os.path.join(dataset_loc, 'labels')
                os.makedirs(label_loc, exist_ok=True)

                print('Downloading your dataset...')
                with Pool(8) as p:
                    if as_json:
                        inputs = zip(self.samples, repeat(dataset_loc))
                    else:
                        inputs = zip(self.samples, repeat(dataset_loc), repeat(False), repeat(self.annotation_type))
                    r = p.starmap(Sample.pull_label, tqdm(inputs, total=len(self.samples)))

                return dataset_loc
            
            else:
                image_loc = os.path.join(dataset_loc, 'images')
                label_loc = os.path.join(dataset_loc, 'labels')

                os.makedirs(image_loc, exist_ok=True)
                os.makedirs(label_loc, exist_ok=True)

                print('Downloading your dataset...')
                with Pool(8) as p:
                    if as_json:
                        inputs = zip(self.samples, repeat(dataset_loc))
                    else:
                        inputs = zip(self.samples, repeat(dataset_loc), repeat(False), repeat(self.annotation_type))
                    r = p.starmap(Sample.pull, tqdm(inputs, total=len(self.samples)))

                return dataset_loc

    def push(self, image_list, label_list=None):
        
        print('Uploading data')
        with Pool(8) as p:
            if label_list:
                inputs = zip(repeat(self), image_list, label_list)
            else:
                inputs = zip(repeat(self), image_list)
            r = p.starmap(Dataset.create_sample, tqdm(inputs, total=len(image_list)))

    def load_samples(self, labeled_only=False, approved_only=False):
        samples = []
        
        filter = ''
        if approved_only:
            filter += '&approved=True'
        if labeled_only:
            filter +='&labeled=True'
        
        page=1
        while(True):
            try:
                endpoint = f'/datasets/{self.uuid}/samples/?page={page}'
                endpoint += filter
                for sample in self.client.get(endpoint):
                    samples.append(Sample(self.client, **sample))
                page+=1
            except:
                break
        
        self.samples = samples

    def create_sample(self, image_file, label_file=None, tags={}):
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
            'asset': asset_data['uuid'],
            'tags': tags
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

    def create_dataset(self, name, description='', categories=[{'id': 1, 'name': 'object'}] ,annotation_type='BBOX'):
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

    def get_or_create_dataset(self, name, owner=None, **kwargs):
        try:
            dataset = self.get_dataset_by_name(name, owner)
        except:
            dataset = self.create_dataset(name, **kwargs)

        return dataset

    def get_model_by_name(self, name, owner=None):
        owner = owner if owner else self.username
        model_data = self.get(f'/models/?name={name}&owner={owner}')
        
        if len(model_data) == 0:
            raise Exception(f'Did not found model "{name}" for owner "{owner}". If this is a shared or public model, please specify the owner')

        return MLModel(self, **model_data[0])


    def create_model(self, name, description='', dataset=None, categories=[] ,annotation_type='BBOX', architecture=''):
        payload = {
            'name': name,
            'description': description,
            'dataset': dataset,
            'categories': categories,
            'annotation_type': annotation_type,
            'architecture': architecture
        }

        model_data = self.post('/models/', payload)

        return MLModel(self, **model_data)

    def get_or_create_model(self, name, owner=None, **kwargs):
        try:
            model = self.get_model_by_name(name,owner=owner)
        except:
            model = self.create_model(name, **kwargs)
        
        return model

