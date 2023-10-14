## Technique used -> CNNS for Color Constancy.

### Method:
### What is color constancy?
It is the process of estimating the scene illuminant and the then correcting the Image based on it's 
estimate to generate a new image of the scene as if it was taken under a reference light source.

### Data Preprocessing:
* Resize the Images to 1200 X 1200.
* Perform contrast normalization using histogram streching.
* Find the corner positions of patches where each patch is of size 32 X 32.
* Mask the patches that has MCC color checker.
* Extract the top-100 brightest patches per image.
* Normalize the patches.

We the use CNN to estimate illuminant of each patch and combine the path scores to obtain an illuminant estimation for the image.

### CNN Architecture
* The Proposed network consists of 5 layers.
* Layer-1: Input   -> 32 X 32 X 3 Dimensional Image patch.
* Layer-2: CONV    -> 240(1 * 1 * 3) kernels, Stride 1 -> 32 X 32 X 240.
* Layer-3: Maxpool -> K = 8 X 8, Stride 8 -> 4 X 4 X 240.
* Layer-4: Flatten the 4 X 4 X 240 => 3840 => FFN with 40 neurons.
* Layer-5: A simple Linear regressio layer with a three dimensional output.


### Dataset used:
* Shi-Gehler Preprocessed Dataset.

### Learning process:
* Train the Model in a 3-Fold cross validation setting.
* Assign illuminant ground truth to each patch associated to the Image which it belongs.
* Using the Euclidean loss as loss function in this process.
* Once the training is complete get the predictions for each patch in the test set.
* we need to take mean or median of predictions of all the patches belonging to a particular Image.
* Then we apply Von-Kries Models to get the corrected Images. 

### Codebase-structure
**Models.py:** Contains the code to build the Models used in the task.
**Data_preparation.py** Contains the code to preprocess and prepare the data for training.
**train.py** Contains the code to train the model.
**utils.py** Contains all the helper functions used in the project.
**visualResults.ipynb** Contains the code and visualizations of producing and produced results.

### Instructions to Execute the code.
* Install conda and create a new virtual env
* Install the required Packages.
* In terminal: python train.py (This completes data prep and trains the model)
* Run the cells in visualResults.ipynb file to get the visualizations of results.