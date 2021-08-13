# importing needed functions
import numpy as np
import scipy.special  # scipy.special for the sigmoid function expit()

# NEURAL NETWORK CLASS DEFINITION
class neuralNetwork:
    # the class consist of three main components: the initialisation, the training, and the query

    # first component of the network
     # INITIALISE THE NEURAL NETWORK
     def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

         # setting placeholders nodes in each input, hidden, and output layer
         # these three make up the head structure of the network, the different layers
         self.inodes = inputnodes
         self.hnodes = hiddennodes
         self.onodes = outputnodes

         # the following part of code creates the matrices for the connections between the layers in the network
         # link weight matrices, wih and who from input layer to hidden layer, and from hidden layer to output layer

         # input to hidden layer
         self.wih = np.random.normal(0.0, pow(self.inodes, -0.5),
        (self.hnodes, self.inodes))

         # hidden to output layer
         self.who = np.random.normal(0.0, pow(self.hnodes, -0.5),
        (self.onodes, self.hnodes))

         # setting placeholder for the learning rate
         self.lr = learningrate

         # activation function is the sigmoid function that is used in the network
         # lamda is used as a quick way to create a function
         self.activation_function = lambda x: scipy.special.expit(x)

         pass

    # second component of the network
     # TRAIN THE NEURAL NETWORK
     def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        # making use of the matrix wih, from input layer to hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer using the sigmoid function
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        # making use of the matrix who, from the hidden layer to the output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer using the sigmoid function
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        # the smaller the difference, the better
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        # transposing means that the column of the outputs becomes a row with the outputs
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

        pass

    # third component of the network
     # QUERY THE NEURAL NETWORK

     def query(self, inputs_list):

         # convert inputs list to 2d array
         # to make sure the input is the same as for the training
         inputs = np.array(inputs_list, ndmin=2).T

         # calculate signals into hidden layer
         # again making use of the wih matrix
         hidden_inputs = np.dot(self.wih, inputs)

         # calculate the signals emerging from hidden layer
         hidden_outputs = self.activation_function(hidden_inputs)

         # calculate signals into final output layer
         # making use of the who matrix
         final_inputs = np.dot(self.who, hidden_outputs)

         # calculate the signals emerging from final output layer
         # the final output is the node that fired. This corresponds to the possible outcomes/labels that you gave as input
         # the output is an array with the possibility for each node. The highest probability is the final output
         # for classification, this means that the
         final_outputs = self.activation_function(final_inputs)

         return final_outputs




