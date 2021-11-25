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

    def test_list_projects(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        projects = client.list_projects()
        assert len(projects) == 2

    def test_get_project_by_uuid(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        project = client.get_project_by_uuid('cb22fe84-aa17-4740-870f-58b2369ba85e')
        assert project.name == 'project_one'

    def test_get_project_by_name(self):
        client = Client(os.environ.get('ML4V_APIKEY'))
        project = client.get_project_by_name('project_one')
        assert project.uuid == 'cb22fe84-aa17-4740-870f-58b2369ba85e'

    def test_create_project(self):
        client = Client(os.environ.get('ML4V_APIKEY'))

        name = 'project_tbd'
        kwargs = dict(
            description='thi is project_tbd',
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
        project = client.create_project(name, **kwargs)
        
        assert project.name == name
        assert project.description == kwargs['description']
        assert project.categories == kwargs['categories']
        assert project.annotation_type == kwargs['annotation_type']
        
        project.delete()