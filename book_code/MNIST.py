import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# scipy.ndimage for rotating image arrays
import scipy.ndimage

#importing the neuralNetwork class from another file
from code.neuralnetwork import neuralNetwork as neuralNetwork

# loading the training data
with open("mnist_train.csv", 'r') as infile:
    data_list = infile.readlines()

# EXAMPLE FOR THE FIRST ENTRY IN THE MNIST_TRAIN DATA SET
# taking all values for the first value in the dataset (target + pixel values)
all_values = data_list[0].split(',')

# reshaping the data to the desired input of 28x28 (= n of pixels)
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))

# changed the code from the one in the book since I am coding in pycharm
# extra line needed to show the plot
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

# INITIALISATION OF THE PARAMETERS FOR THE NEURAL NETWORK
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 5

# printing updates on where the code is running
print('starting with training the network')

for e in range(epochs):
    print('now training epoch', e)

 # go through all records in the training data set
    for record in data_list:

 # split the record by the ',' commas
        all_values = record.split(',')

         # scaling the input to a range within 0.01 and 0.99 to avoid killing or saturating the nodes
         # in the network. Starting from the second value in the vector since the first value is the
         # 'gold' label.
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # create the target output values, all the lowest possible value of 0.01
        targets = numpy.zeros(output_nodes) + 0.01

        # the first value in the vector is the 'gold' label, set this to 0.99 in the targets list
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)

        # TODO: OUTCOMMENT FOLLOWING SECTION TO TRAIN WITH ROTATIONS
        ## create rotated variations
        # rotated anticlockwise by x degrees
        # inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
        # n.train(inputs_plusx_img.reshape(784), targets)
        #
        # # rotated clockwise by x degrees
        # inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
        # n.train(inputs_minusx_img.reshape(784), targets)
        pass
    pass

# loading the training data
with open("mnist_test.csv", 'r') as infile:
    test_data_list = infile.readlines()

# test the neural network
print('testing the neural network')

# empty lists for saving gold labels and predictions
gold = []
predicted = []

# go through all the records in the test data set
for record in test_data_list:

    # split the record by the ',' commas
    all_values = record.split(',')

    # correct answer is first value
    correct_label = int(all_values[0])
    gold.append(correct_label)

    # scale and shift the inputs, same range as for training
    # between 0.01 and 0.99 to avoid killing nodes or saturating the network
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # query the network
    outputs = n.query(inputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    predicted.append(label)

    pass

# print classification report
print(classification_report(gold, predicted))




