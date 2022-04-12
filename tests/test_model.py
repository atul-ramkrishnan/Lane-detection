import os
import sys
import inspect
import torch
currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import model



def test_VGG16_SegNet():
    model_ = model.VGG16_SegNet()
    input = torch.ones([100, 3, 128, 256])
    output = model_(input)
    assert output.shape == (
        100, 2, 128, 256), "VGG16_SegNet output has incorrect shape"


def test_VGG16_SegNet_pretrained():
    model_ = model.VGG16_SegNet_pretrained()
    input = torch.ones([100, 3, 128, 256])
    output = model_(input)
    assert output.shape == (
        100, 2, 128, 256), "VGG16_SegNet_pretrained output has incorrect shape"


def test_VGG11_SegNet_pretrained():
    model_ = model.VGG11_SegNet_pretrained()
    input = torch.ones([100, 3, 128, 256])
    output = model_(input)
    assert output.shape == (
        100, 2, 128, 256), "VGG11_SegNet_pretrained output has incorrect shape"


def test_VGG11_SegNet_ConvLSTM_pretrained():
    model_ = model.VGG11_SegNet_ConvLSTM_pretrained()
    input = torch.ones([100, 5, 3, 128, 256])
    output = model_(input)
    assert output.shape == (
        100, 2, 128, 256),\
        "VGG11_SegNet_ConvLSTM_pretrained output has incorrect shape"
