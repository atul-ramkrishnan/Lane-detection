"""
Contains a dict of the various models implemented.
"""

import model

models = {
    "VGG16_SegNet": model.VGG16_SegNet(),
    "VGG16_SegNet_pretrained": model.VGG16_SegNet_pretrained(),
    "VGG11_SegNet_pretrained": model.VGG11_SegNet_pretrained(),
    "VGG11_SegNet_ConvLSTM_pretrained":
    model.VGG11_SegNet_ConvLSTM_pretrained()
}
