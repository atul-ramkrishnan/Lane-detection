Guide to run the Notebook

Prerequisites
The dataset is available at -- https://drive.google.com/drive/folders/1HWVEEQMefz1nlmxbjhKJinBpRwp4QVC7?usp=sharing
There are two folders --
1. data - contains all the training / validation / testing data
2. pretrained - contains all the pretrained model weights for the two models I compared. The pretrained weights are available for various sizes of the training set on which it has been trained ranging from 1000 to 17000.

The notebook is arranged as follows --
1. Imports -- Imports necessary packages
2. Config / Globals -- Defines constants, file paths, sets seeds, etc
3. Dataloader -- Dataloader class
4. EarlyStopping -- EarlyStopping class
5. Models -- Defines all the models
6. Training / Validation functions -- Defines the train() and val() functions
7. Testing functions -- Defines the get_prediction() and get_evaluation_scores() functions
8. Instantiating Dataloaders -- Creates dataloader objects for all the experiments below this section
9. Data exploration -- A few inputs and their targets
10. Training
	10. A. Training the CNN-only model
	10. B. Training the ConvLSTM model
11. Testing
	11. A. Testing the CNN-only model
	11. B. Testing the ConvLSTM model
12. Side by side comparision of the challenging example as mentioned in the report
13. Plots performance against number of samples trained on.
