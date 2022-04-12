# Lane detection using CNN-RNN architecture

## Prerequisites
The dataset for the project is available on [Google drive.](https://drive.google.com/drive/folders/1HWVEEQMefz1nlmxbjhKJinBpRwp4QVC7?usp=sharing)
The drive folder also contains weights for the models I've trained at various sizes of the training data from 1000 to 17000 samples at an interval of 1000 samples.
There is also a data.zip that has the same dataset but in a zip format (making it easy to unzip it on Google Colab).

## Repository organization
config.py -- Contains configuration options like the paths to the training data, which model to train, etc.

data.py -- Implements the dataloader.

model.py -- Contains implementations of all the models.

model_dispatcher.py -- Maintains a dict of all the models in models.py

early_stopping.py -- Implements the early stopping mechanism.

notebook -- Contains the jupyter notebook along with a README.

tests -- Contains pytest testcases for a few functions.
