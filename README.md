Code from "A study on multi-exponential inversion of nuclear magnetic resonance relaxation data with deep learning"
=======================================================================================================================
This project is under review.
The key programs containing NMR forward simulation process and neural network structure will be published first. 
Other programs need to add comments and optimize the code architecture incrementally. 
Therefore, this is a continuous process of revision and publication.

Requirements
------------------------------------------------------
Here is a list of libraries you might need to install to execute the code:
* python (=3.9.7)
* tensorflow-gpu (=2.7.0)
* numpy (=1.21.4)
* pandas (=1.3.4)
* matplotlib (=3.4.3)

Computer Device
------------------------------------------------------
All programs run under the server with 2.90 GHz Intel Core i7-10700 and NVIDIA GeForce RTX 3060.

Data
------------------------------------------------------
The function 'data_generated' will used to generated simulated datset.
Run the 'main.py' will finish simulated data generated, model training, and the prediction of test set.
Please make sure you have enough CPU memory and GPU computing power before running, which may take a long time.