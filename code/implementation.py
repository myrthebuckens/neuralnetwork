import pandas as pd
import gensim
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import argparse
import itertools

#importing the neuralNetwork class from another file
from code.neuralnetwork import neuralNetwork as neuralNetwork

def extracting_embeddings(data_path, language_model):
    """
    Function to extract embedding representations for the tokens in the data from
    the GoogleNews-negative300 pretrained embedding model
    :param data_path: path to data set
    :param embedding_model: path to embedding model
    :return: list with embedding representations, list with targets
    """
    # empty list to save
    embeddings = []

    # reading data as pandas dataframe
    data = pd.read_csv(data_path, sep = '\t', header = 0)

    # indicating which column contains which type of data
    tokens = data['Token']

    # in the preprocessing script, I added a header name for this column (NER), but every time I ran it, it got changed to '0'
    labels = data['0']

    # get embedding representation from the language model
    for token in tokens:
            if token in language_model:
                vector = language_model[token]
            else:
                vector = [0] * 300

            # adding representation to list
            embeddings.append(vector)

    # find out range of embedding values in vectors
    maximum = 0
    for vectors in embeddings:
        max_value = max(vectors)

        if max_value >= maximum:
            maximum = max_value

        else:
            maximum = maximum

    return embeddings, labels

# maximum values in the vectors for training, validation and test set
training = 1.2578125
validation = 1.1953125
test = 1.46875


def training_neural_network(embeddings, labels, hidden_nodes, learning_rate, epochs):
    """
    Function to train the neural network on the data
    :param embeddings: list with vectors from the embedding model, representating every token in the data
    :param labels: list with the gold labels
    :return: n, the trained network
    """

    # setting n of nodes for network
    # input is 300 because of the length of the embeddings
    input_nodes = 300

    # output is 5 because of the number of possible targets
    output_nodes = 5

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes,output_nodes,
                      learning_rate)

    # transforming the labels for NER to numeric labels for the network
    le = preprocessing.LabelEncoder()
    labels_trans = le.fit_transform(labels)

    # train the neural network
    # setting the n of epochs for training
    #epochs = 1

    for e in range(epochs):
        print('training update: now running epoch', e)
        # go through all records in the training data set
        for vector, label in zip(embeddings, labels_trans):

            # 1.46875 is the highest value in the vectors from the embeddings for train, valid and test data
            inputs = (np.asfarray(vector) / 1.46875 * 0.99) + 0.01

            # create array with empty items with same length as n of output nodes
            targets = np.zeros(output_nodes) + 0.01

            # get target value from label and change that value to 0.99
            targets[int(label)] = 0.99

            # training the network
            n.train(inputs, targets)
            pass
        pass

    print('finished training process')

    return n

def running_neural_network(test_path, test_embeddings, test_labels, n):
    """
    Function to run the trained neural network
    :param test_embeddings: the embedding representations for the test data
    :param test_labels: the NER labels from the test data
    :param n: the trained neural network
    :return:
    """
    print('starting with testing the network')
    # transforming the labels for NER to numeric labels for the network
    le = preprocessing.LabelEncoder()
    test_labels_trans = le.fit_transform(test_labels)

    # test the neural network
    # empty list for predictions
    predicted_labels = []

    # looping through every vector and label in the data set
    for vector, label in zip(test_embeddings, test_labels_trans):

        # 1.46875 is the highest value in the vectors from the embeddings for train, valid and test data
        inputs = (np.asfarray(vector) / 1.46875 * 0.99) + 0.01

        # create array with empty items with same length as n of output nodes
        targets = np.zeros(5) + 0.01 #5 for the number of output nodes

        # get target value from label and change that value to 0.99
        original_label = label
        correct_label = targets[int(label)] = 0.99

        # make predictions for every input item
        outputs = n.query(inputs)

        # the index of the highest value corresponds to the label
        predicted_label = np.argmax(outputs)
        predicted_labels.append(predicted_label)

        pass

    # Saving final output to conll file
    with open(test_path) as infile:
        data = pd.read_csv(infile, sep='\t')

    inverted = le.inverse_transform(predicted_labels)

    preds = pd.DataFrame(inverted)
    final_df = pd.concat([data, preds], axis = 1)
    header = ['token', 'POS', 'Phrase', 'NER', 'PRED']
    final_df.to_csv('../data/final_output.conll', sep='\t', header=header)

def main():
    adding arguments to functions
    parser = argparse.ArgumentParser(description='This script extracts embeddings for the data and trains and runs a neural network')

    parser.add_argument('training_path',
                        help='file path to conll file with preprocessed training data. Recommended path: "../data/conll2003/train_converted.conll"')

    parser.add_argument('test_path',
                        help='file path to conll file with preprocessed test data. Recommended path: "../data/conll2003/test_converted.conll"')

    parser.add_argument('word_embedding_model',
                        help='path to word embedding model. Recommended path: "../GoogleNews-vectors-negative300.bin"')

    args = parser.parse_args()


    # loading language model
    word_embedding_model = word_embedding_model
    language_model = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_model, binary=True)

    # extracting features
    embeddings, labels = extracting_embeddings(training_path, language_model)

    # training with different hyper parameters settings for learning rate, hidden nodes and epochs
    # first list in list is learning rate
    # second list in list is hidden nodes
    # third list in list is epochs

    # TODO: TO TRAIN THE NETWORK WITH ALL POSSIBLE HYPER-PARAMETER SETTINGS, OUTCOMMENT THE FOLLOWING LINES
    # all = [[0.1, 0.2, 0.3, 0.4], [50, 150, 200], [1, 2, 5, 7]]

    # all_combinations = (list(itertools.product(*all)))
    #
    # for combinations in all_combinations:
    #     print('learning rate:',combinations[0] , 'hidden nodes:', combinations[1], 'epochs:', combinations[2])
    #
    #     learning_rate = combinations[0]
    #     hidden_nodes = combinations[1]
    #     epochs = combinations[2]

    # TODO: COMMENT OUT THE FOLLOWING THREE LINES OF CODE TO TRAIN THE NETWORK WITH ALL HYPER-PARAMETER SETTINGS
    # BEST HYPER-PARAMETER SETTINGS
    # hidden is random in between input and output n
    hidden_nodes = 150

    # learning rate initialization
    learning_rate = 0.1

    epochs = 7

    # TODO: PLACE THE FOLLOWING THREE LINES OF CODE INSIDE THE LOOP FOR TRAINING WITH ALL HYPER-PARAMETER SETTINGS
    # training network
    n = training_neural_network(embeddings, labels, hidden_nodes, learning_rate, epochs)

    # extracting gold features
    test_embeddings, test_labels = extracting_embeddings(test_path, language_model)

    # running the network
    running_neural_network(test_path, test_embeddings, test_labels, n)


if __name__ == '__main__':
    main()



