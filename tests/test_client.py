import os

from ml4vision import Client

from dotenv import load_dotenv
load_dotenv()

class TestClient:

    def test_create_client(self):
        client = Client(os.environ.get('ML4V_APIKEY'))

    def test_username(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        assert  client.username == os.environ.get('ML4V_USERNAME')

    def test_list_datasets(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        datasets = client.list_datasets()
        assert len(datasets) == 2

    def test_get_dataset_by_uuid(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        dataset = client.get_dataset_by_uuid('cb22fe84-aa17-4740-870f-58b2369ba85e')
        assert dataset.name == 'dataset_one'

    def test_get_dataset_by_name(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        dataset = client.get_dataset_by_name('dataset_one')
        assert dataset.uuid == 'cb22fe84-aa17-4740-870f-58b2369ba85e'

    def test_create_dataset(self):
        client = Client(os.environ.get('ML4V_APIKEY'))

        name = 'dataset_tbd'
        kwargs = dict(
            description='thi is dataset_tbd',
            categories=[{
                'id': 0,
                'name': 'object_0'
            },
            {
                'id': 1,
                'name': 'object_1'
            }],
            annotation_type='BBOX',
        )
        dataset = client.create_dataset(name, **kwargs)
        
        assert dataset.name == name
        assert dataset.description == kwargs['description']
        assert dataset.categories == kwargs['categories']
        assert dataset.annotation_type == kwargs['annotation_type']
        
        dataset.delete()