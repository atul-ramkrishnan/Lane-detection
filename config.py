import torch

"""
Contains various configuration settings.
"""

train_path = "./data/train_index_medium.txt"
val_path = "./data/val_index_small.txt"
demo_path = "./data/test_index_demo.txt"
test_path = './data/test_index_merged.txt'
save_path = "./saved_images/"
# Path to save the model.
pretrained_path = "./pretrained/"


seed = 0
device = torch.device('cpu')
train_batch_size = 32

evaluate_batch_size = 32

predict_batch_size = 1

CNN_only_models = [
    "VGG16_SegNet",
    "VGG16_SegNet_pretrained",
    "VGG11_SegNet_pretrained"]
ConvLSTM_models = ["VGG11_SegNet_ConvLSTM_pretrained"]
model_to_train = "VGG11_SegNet_ConvLSTM_pretrained"
test_mode = "get_evaluation_scores"
model_to_test = "VGG11_SegNet_ConvLSTM_pretrained"
# Name of the pretrained model to test.
test_pretrained_model = "VGG11_SegNet_ConvLSTM_pretrained17000"

num_epochs = 10
learning_rate = 0.001
train_max_samples = None 	# Use None to train on the entire dataset.
val_max_samples = 1000
get_prediction_max_samples = 10
get_evaluation_scores_max_samples = 1000
class_weight = [0.02, 1.02]
