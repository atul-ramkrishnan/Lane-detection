import os
import sys
import inspect
import shutil
import pytest
currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import model_dispatcher
from early_stopping import EarlyStopping

tempDir = './temp'


def create_temp_folder():
    os.mkdir(tempDir)


@pytest.fixture
def cleanup_fixture():
    yield
    try:
        shutil.rmtree(tempDir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


@pytest.mark.usefixtures('cleanup_fixture')
def test_early_stopping():
    create_temp_folder()
    val_loss_list = [0.1, 0.05, 0.002, 0.2, 0.3, 0.4, 0.5]

    early_stopping = EarlyStopping(
        patience=1,
        save_path=tempDir +
        "/test_model",
        min_delta=0.001)

    model = model_dispatcher.models["VGG16_SegNet"]
    for val_loss in val_loss_list:
        if early_stopping(model, val_loss):
            assert val_loss == 0.2, "Early stopping stopped prematurely"
            break
        assert val_loss < 0.3, "Early stopping did not stop"

    early_stopping = EarlyStopping(
        patience=2,
        save_path=tempDir +
        "/test_model",
        min_delta=0.001)

    model = model_dispatcher.models["VGG16_SegNet"]
    for val_loss in val_loss_list:
        if early_stopping(model, val_loss):
            assert val_loss == 0.3, "Early stopping stopped prematurely"
            break
        assert val_loss < 0.4, "Early stopping did not stop"
