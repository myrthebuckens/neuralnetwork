# Neural Network
Repository for the tutorial on neural networks, as part of the course ML for NLP at the Vrije Universiteit (2020/2021)
In this project, there is made use of the GoogleNews-negative300.bin embedding model. This model needs to be installed on your machine and the path to this location can be entered as an argument for the implementation script. 
The different data that is used are the conll2003 data sets, which are uploaded in this github repository for your convenience, and the mnist_test and mnist_train datasets. The latter can be obtained from http://www.pjreddie.com/media/files/mnist_train.csv (training)
and http://www.pjreddie.com/media/files/mnist_test.csv (testing) and saved in the book_code folder, for easy running of the MNIST.py script.

### Requirements 
The needed requirements can be found in `requirements` and installed by running
```pip install requirements``` from your terminal.

### book_code
This folder contains code that was written along with part 2 and 3 of the book 'Make Your Own Neural Network' by Tariq Rashid (2016).
* `MNIST.py`: the code to train and test the neural network on the MNIST data set. 

### code 
This folder contains code for the implementation of the neural network for Named Entity Recognition
* `__init__.py`: initialised script for importing the neural network into implementation
* `evaluation.py`: script for evaluating the predictions of the trained neural network
* `implementation.py`: script that contains functions for training the neural network with multiple hyper-parameter settings (optional, see TODO comments for running with different parameter settings). The standard settings as in the script are the ultimate settings from the testing phase. The predictions from the model are saved to '../data/final_output.conll'
* `neuralnetwork.py`: script containing the class for the neural network, after Rashid (2016) including personal comments 
* `preprocessing.py`: script for preprocessing the conll data from the SharedTask 2003 on Named Entity Recognition (online available)

### data
This folder contains data for the Named Entity Recognition study
* `final_output.conll`: conll file with the test data and the predictions by the best settings from the neural network

#### conll2003
This subfolder contains the original test, train and validation conll files, and the preprocessed files (ending on _converted). In this subfolder, there is also a metadata file with information on the original data files.

