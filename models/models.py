from .attention_modules import SE, BAMchannel, BAMspatial, BAM, CBAMspatial, CBAMchannel, CBAM
from .attention_resnet import resnet34, resnet50


ATTENTION_TYPES = ['None', 'SE', 'BAM', 'CBAM', 'BAMchannel', 'BAMspatial', 'CBAMchannel', 'CBAMspatial']
TYPES_TO_MODULE = {'SE': SE,
                   'BAM': BAM, 'BAMspatial': BAMspatial, 'BAMchannel': BAMchannel,
                   'CBAM': CBAM, 'CBAMspatial': CBAMspatial, 'CBAMchannel': CBAMchannel}


def create_attention_dict(attention):
    assert attention in ATTENTION_TYPES, 'Wrong attention type: {}. '.format(attention)
    if attention != 'None':
        attention_module = TYPES_TO_MODULE[attention]
        attention_dict = {'type': attention_module}
    else:
        attention_dict = None
    return attention_dict


def create_model(arch, attention, num_classes=100):
    if arch == 'resnet34':
        network = resnet34
    else:
        network = resnet50
    attention_dict = create_attention_dict(attention)

    model = network(False, num_classes=100, attention_dict=attention_dict)
    return model
