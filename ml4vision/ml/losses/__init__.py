from torch import nn
from .bcedice import BCEDiceLoss
from .centernet import centernet_loss


def get_loss(name, loss_kwargs={}):
    if name == 'bce':
        return nn.BCEWithLogitsLoss(**loss_kwargs)
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss(**loss_kwargs)
    elif name == 'bcedice':
        return BCEDiceLoss(**loss_kwargs)
    elif name == 'centernet':
        return centernet_loss
    else:
        raise RuntimeError(f'Loss {name} is not available!')