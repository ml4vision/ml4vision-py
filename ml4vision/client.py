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
import yaml

class MLModel:

    def __init__(self, client, **kwargs):
        self.client = client
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_version(self, model_file, params={}, metrics={}):

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
            'params': params,
            'metrics': metrics
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

    def pull_label(self, location='./', type='BBOX', format=None):
        self.load_label()

        if type=='BBOX':

            if format == 'yolo':
                asset_filename = self.asset['filename']
                label_filename = os.path.splitext(asset_filename)[0] + '.txt'
                label_location = os.path.join(location, 'labels', label_filename)

                im_w, im_h = self.asset['metadata']['size'] 

                with open(label_location, 'w') as f:
                    for l in self.label['annotations']:
                        cl = l['category_id'] - 1
                        x_min, y_min, w, h = l['bbox']
                        x_max = x_min + w
                        y_max = y_min + h

                        x_min = min(max(0, x_min), im_w)
                        y_min = min(max(0, y_min), im_h)
                        x_max = min(max(0, x_max), im_w)
                        y_max = min(max(0, y_max), im_h)

                        xc, yc = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                        w, h = x_max - x_min, y_max - y_min
                        
                        xc, yc = xc / im_w, yc / im_h
                        w, h = w / im_w, h / im_h

                        f.write(f'{cl} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}\n')

            else:

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

    def pull(self, location='./', type='BBOX', format=None):
        self.pull_image(location=location)
        self.pull_label(location=location, type=type, format=format)

    def delete(self):
        self.client.delete(f'/samples/{self.uuid}/')

class Project:
    
    def __init__(self, client, **project_data):
        self.client = client
        for key, value in project_data.items():
            setattr(self, key, value)

    def pull(self, location='./', format=None):
        project_loc = os.path.join(location, self.name)

        if format == 'coco':
            image_loc = os.path.join(project_loc, 'images')
            os.makedirs(image_loc, exist_ok=True)
            
            print('Loading images')
            with Pool(8) as p:
                inputs = zip(self.samples, repeat(project_loc))
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

            with open(f'{os.path.join(project_loc, self.name)}.json', 'w') as f:
                coco = {
                    'images': images,
                    'categories': self.categories,
                    'annotations': annotations
                }
                json.dump(coco, f)

        elif format == 'yolo':
            image_loc = os.path.join(project_loc, 'images')
            label_loc = os.path.join(project_loc, 'labels')

            os.makedirs(image_loc, exist_ok=True)
            os.makedirs(label_loc, exist_ok=True)

            print('Downloading your project...')
            with Pool(8) as p:
                inputs = zip(self.samples, repeat(project_loc), repeat(self.annotation_type), repeat('yolo'))
                r = p.starmap(Sample.pull, tqdm(inputs, total=len(self.samples)))

            # create train-val txt file
            with open(os.path.join(project_loc, 'train.txt'), 'w') as f:
                for s in self.samples:
                    if s.split == 'TRAIN':
                        f.write(f'./images/{s.asset["filename"]}\n')

            with open(os.path.join(project_loc, 'val.txt'), 'w') as f:
                for s in self.samples:
                    if s.split == 'VAL':
                        f.write(f'./images/{s.asset["filename"]}\n')

            # create dataset yaml file
            with open(os.path.join(project_loc, 'dataset.yaml'), 'w') as f:
                yaml_content = {
                    'path': os.path.abspath(project_loc),
                    'train': 'train.txt',
                    'val': 'val.txt',
                    'names': {cat['id']-1:cat['name'] for cat in self.categories}
                }
                yaml.dump(yaml_content, f)

        else:
            image_loc = os.path.join(project_loc, 'images')
            label_loc = os.path.join(project_loc, 'labels')

            os.makedirs(image_loc, exist_ok=True)
            os.makedirs(label_loc, exist_ok=True)

            print('Downloading your project...')
            with Pool(8) as p:
                inputs = zip(self.samples, repeat(project_loc), repeat(self.annotation_type))
                r = p.starmap(Sample.pull, tqdm(inputs, total=len(self.samples)))

            return project_loc

    def push(self, image_list, label_list=None):
        
        print('Uploading data')
        with Pool(8) as p:
            if label_list:
                inputs = zip(repeat(self), image_list, label_list)
            else:
                inputs = zip(repeat(self), image_list)
            r = p.starmap(Project.create_sample, tqdm(inputs, total=len(image_list)))

    def load_samples(self, labeled_only=False, approved_only=False, split=None):
        samples = []
        
        filter = ''
        if approved_only:
            filter += '&approved=True'
        if labeled_only:
            filter +='&labeled=True'
        if split is not None:
            filter +=f'&split={split}'
        
        page=1
        while(True):
            try:
                endpoint = f'/projects/{self.uuid}/samples/?page={page}'
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
        sample_data = self.client.post(f'/projects/{self.uuid}/samples/', payload)

        sample = Sample(client=self.client, **sample_data)

        # upload label
        if label_file:
            label = np.array(Image.open(label_file))
            annotations = mask_utils.label_to_annotations(label)
            sample.update_label(annotations)

        return sample

    def make_split(self):
        return self.client.put(f'/projects/{self.uuid}/make_split/')

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

    def patch(self, endpoint, payload={}):
        
        response = requests.patch(self.url + endpoint, json=payload, headers = self._get_headers())

        if response.status_code == 200:
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
            raise Exception(f'Did not found project "{name}" for owner "{owner}". If this is a shared or public project, please specify the owner')

        return Project(self, **project_data[0])

    def create_project(self, name, description='', categories=[{'id': 1, 'name': 'object'}] ,annotation_type='BBOX'):
        payload = {
            'name': name,
            'description': description,
        }
        if categories:
            payload['categories'] = categories
        if annotation_type:
            payload['annotation_type'] = annotation_type

        project_data = self.post('/projects/', payload)
        
        return Project(self, **project_data)

    def get_or_create_project(self, name, owner=None, **kwargs):
        try:
            project = self.get_project_by_name(name, owner)
        except:
            project = self.create_project(name, **kwargs)

        return project

    def get_model_by_name(self, name, owner=None):
        owner = owner if owner else self.username
        model_data = self.get(f'/models/?name={name}&owner={owner}')
        
        if len(model_data) == 0:
            raise Exception(f'Did not found model "{name}" for owner "{owner}". If this is a shared or public model, please specify the owner')

        return MLModel(self, **model_data[0])

    def create_model(self, name, description='', project=None, categories=[] ,annotation_type='BBOX', architecture=''):
        payload = {
            'name': name,
            'description': description,
            'project': project,
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

