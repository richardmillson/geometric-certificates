from . import datasets, models
from .models.bninception import bninception
from .models.cafferesnet import cafferesnet101
from .models.dpn import dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131
# to support pretrainedmodels.__dict__['nasnetalarge']
# but depreciated
from .models.fbresnet import fbresnet152
from .models.inceptionresnetv2 import inceptionresnetv2
from .models.inceptionv4 import inceptionv4
from .models.nasnet import nasnetalarge
from .models.nasnet_mobile import nasnetamobile
from .models.resnext import resnext101_32x4d, resnext101_64x4d
from .models.senet import (se_resnet50, se_resnet101, se_resnet152,
                           se_resnext50_32x4d, se_resnext101_32x4d, senet154)
from .models.torchvision_models import (alexnet, densenet121, densenet161,
                                        densenet169, densenet201, inceptionv3,
                                        resnet18, resnet34, resnet50,
                                        resnet101, resnet152, squeezenet1_0,
                                        squeezenet1_1, vgg11, vgg11_bn, vgg13,
                                        vgg13_bn, vgg16, vgg16_bn, vgg19,
                                        vgg19_bn)
from .models.utils import model_names, pretrained_settings
from .models.xception import xception
from .version import __version__
