from .segm_config import get_segm_config
from .det_config import get_det_config

def get_config(client, project_name, project_owner=None):

    # load project info
    project = client.get_project_by_name(project_name, owner=project_owner)

    if project.annotation_type == 'SEGMENTATION':
        return get_segm_config(project)
    elif project.annotation_type == 'BBOX':
        return get_det_config(project)
    else:
        raise RuntimeError(f'Trainer not implemented for project of annotation type: {project.annotation_type}.')
