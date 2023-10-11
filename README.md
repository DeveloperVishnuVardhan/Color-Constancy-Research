## Techniques used -> 1. CNNS for Color Constancy.

### Method-1
#### What is color constancy?
It is the process of estimating the scene illuminant and the then correcting the Image based on it's 
estimate to generate a new image of the scene as if it was taken under a reference light source.

#### Data Preprocessing:
* Sample Non-Overlapping patches
* Perform Contrast Normalization for each sample using Histogram Streching.

We the use CNN to estimate illuminant of each patch and combine the path scores to obtain an illuminant estimation for the image.

#### CNN Architecture
* The Proposed network consists of 5 layers.
* Layer-1: Input   -> 32 X 32 X 3 Dimensional Image patch.
* Layer-2: CONV    -> 240(1 * 1 * 3) kernels, Stride 1 -> 32 X 32 X 240.
* Layer-3: Maxpool -> K = 8 X 8, Stride 8 -> 4 X 4 X 240.
* Layer-4: Flatten the 4 X 4 X 240 => 3840 => FFN with 40 neurons.
* Layer-5: A simple Linear regressio layer with a three dimensional output.


#### Dataset used:
* Shi-Gehler and SFU subset.

#### Learning process:
* Resize the Images to 1200 X 1200.
* Extracting non-overlapping patches each of size 32 X 32.
* Perform contrast normalization using histogram streching.
* Train the Model in a 3-Fold cross validation setting.
* Assign illuminant ground truth to each patch associated to the Image which it belongs.(**Which we need to Figure out**)
* Using the Euclidean loss as loss function in this process.
* Fine tune the Model using Angular loss by adding knowledge about the way local estimates are 
   pooled to generate a single global estimate for each image. (**Need to be figured out**)

#### Preprocessing steps:
* Images in the Shi-Gehler dataset are resized to 1200 X 1200 and MCC is removed from every image.
* The top 100 Brightest patches from each Image are extracted.


### Codebase-structure
**Models.py:** Contains the code to build the Models used in the task.
