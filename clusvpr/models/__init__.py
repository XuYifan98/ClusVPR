from __future__ import absolute_import
from .vgg import *
from .clusvpr import *


__factory = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'clusvpr': ClusVPR,
    'embednet': EmbedNet,
    'embedregiontrans': EmbedRegionTrans,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'mobilenetv3_large', 'mobilenetv3_small', 'transgvlad',
        'embednet', 'embedregiontrans'.
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
