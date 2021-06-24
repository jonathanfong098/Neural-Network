"""
Assignment Five
   Jonathan Fong
   8/4/2020
   Assignment 1: This program setups up NNData to manage and test our neural network data. The program then tests to see
                 if the NNData class is setup properly.
   Assignment 2 Check In: This program adds new attributes (_train_indices, test_indices, _train_pool, and _test_pool)
                       and it adds a new method (split_set()) to the NNData class. Then the program tests to see if
                       the additions to the NNData class are working properly.
   Assignment 2: This program adds four new methods, prime_data(), get_one_item(), number_of_samples(), and
                  pool_is_empty(), to the NNData class. Then the program tests to see if the the additions to the
                  NNData class are working properly.
   Assignment 3 Check In 1: This program setups up two new classes MultiLinkNode, a parent class for neurodes, and
                          Neurode, a subclass of MultiLinkNode. The program then tests to see if the two classes are
                          setup correctly.
   Assignment 3 Check In 2: This program setups a new class called FFNeurode. The program then tests to see if the two
                            classes are setup correctly.
   Assignment 3 : This program setups two new classes BPNeurode and FFBPNeurode. The program then tests to see if the
                  two classes are setup correctly.
   Assignment 4 Check In 1: This program setups up two new classes Node and DoublyLinkedList. The program then tests to
                            see if the two classes are setup correctly.
   Assignment 4 : This program setups up a new class LayerList. The program then tests to see if the LayerList class
                  is setup correctly.
   Assignment 5 Check In 1: This program setups a new class called FFBPNetwork. The program then tests to see if
                            the FFBPNetwork is setup correctly.
   Assignment 5: This program adds a new method test(). The program then tests to see if FFBPNetwork is setup correctly.
"""
from Assignment_Two import NNData
from Assignment_Four import LayerList
import math
from matplotlib import pyplot as plt
import numpy as np

# variables used for matplotib graph
test_y = []
test_x = []


class FFBPNetwork:
    """
    This class that setups an FFBPNetwork.

    Attributes:
        layer list (LayerList) : new layer list object
        input nodes count (integer) : number of input neurodes
        output nodes count (integer) : number of output neurodes
    """
    class EmptySetExcpetion(Exception):
        """
        This class is a user define exception class and is called when the LayerList is empty.
        """
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        """
        The constructor for FFBPNetwork.

        Parameters:
            num inputs (integer) : number of input neurodes
            num outputs (integer) : number of output neurodes
        """
        self._layer_list = LayerList(num_inputs, num_outputs)

        self._input_nodes_count = num_inputs
        self._output_nodes_count = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        """
        This function adds a hidden layer to the FFBPNetwork

        Parameters:
            num nodes (integer) : number of hidden neurodes
            position (integer) : the position to add the hidden layers after
        """
        if type(num_nodes) == int:
            self.layer_list.reset_to_head()

            for position in range(position):
                self.layer_list.move_forward()
            self.layer_list.add_layer(num_nodes)

            self.layer_list.reset_to_head()
        else:
            raise TypeError('You must pass an integer for the number of hidden neurodes')

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.RANDOM):
        """
        This function trains the FFBPNetwork.

        Parameters:
            data set (NNData) : data set that trains network
            epochs (integer) : number of times the network is trained
            verbosity (integer) : indicates what the network should report
            order (NNData.Order) : indicates whether the training data is presented in sequential or random order
        """
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetExcpetion('The training set is empty')
        else:
            rmse_value = 0

            for epoch in range(epochs):
                data_set.prime_data(NNData.Set.TRAIN, order)

                # stores all the errors of one training epoch
                training_errors = []

                while not data_set.pool_is_empty(NNData.Set.TRAIN):
                    # get a feature and label pair from dataset
                    training_feature, training_label = data_set.get_one_item(NNData.Set.TRAIN)

                    # give features to input nodes as input values
                    for index, input_node in enumerate(self.layer_list.input_nodes):
                        input_node.set_input(training_feature[index])

                    # check values of output nodes and calculates root-mean-square error
                    for index, output_node in enumerate(self.layer_list.output_nodes):
                        training_errors.append((float(training_label[index]) - float(output_node.value)) ** 2)

                    # gives labels to output nodes as expected values
                    for index, output_node in enumerate(self.layer_list.output_nodes):
                        output_node.set_expected(training_label[index])

                    # present report
                    output_nodes_value = [output_node.value for output_node in self.layer_list.output_nodes]
                    if (verbosity > 1) and (epoch % 1000 == 0):
                        print(f'Training Information\n'
                              f'-------------------------\n'
                              f'input values: {training_feature}\n'
                              f'expected values: {training_label}\n'
                              f'output values: {output_nodes_value}\n')

                # finds the root-mean-square value of one training epoch
                rmse_value = math.sqrt(sum(training_errors) / len(training_errors))
                # present report
                if (verbosity > 0) and (epoch % 100 == 0):
                    print(f'Epoch {epoch} root-mean-square error: {rmse_value}\n')

            print(f'Final root-mean-square error: {rmse_value}\n')

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """
        This function test the FFBPNetwork.

        Parameters:
            data set (NNData) : data set that network tests
            order (NNData.Order) : indicates whether the training data is presented in sequential or random order
        """
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetExcpetion('The testing set is empty')
        else:
            data_set.prime_data(NNData.Set.TEST, order)

            # stores all the errors of a test run
            testing_errors = []

            while not data_set.pool_is_empty(NNData.Set.TEST):
                # get a feature and label pair from dataset
                testing_feature, testing_label = data_set.get_one_item(NNData.Set.TEST)

                # give features to input nodes as input values
                for index, input_node in enumerate(self.layer_list.input_nodes):
                    input_node.set_input(testing_feature[index])

                # check values of output nodes and calculates root-mean-square error
                for index, output_node in enumerate(self.layer_list.output_nodes):
                    testing_errors.append((float(testing_label[index]) - float(output_node.value)) ** 2)

                # gives labels to output nodes as expected values
                for index, output_node in enumerate(self.layer_list.output_nodes):
                    output_node.set_expected(testing_label[index])

                # present report
                output_nodes_value = [output_node.value for output_node in self.layer_list.output_nodes]
                print(f'Testing Information\n'
                      f'------------------------\n'
                      f'input values: {testing_feature}\n'
                      f'expected values: {testing_label}\n'
                      f'ouput values: {output_nodes_value}\n')

                # uses features and values for output node for matplotib graphs
                test_x.append(testing_feature)
                test_y.append(output_nodes_value)

            # finds the root-mean-square value a testing run
            rmse_value = math.sqrt(sum(testing_errors) / len(testing_errors))
            print(f'Final root-mean-square error: {rmse_value}\n')

    @property
    def layer_list(self):
        """
        This property returns layer list.

        Returns:
            layer list (LayerList) : layer list object
        """
        return self._layer_list


def run_iris():
    """
    This function uses the iris data set to train and test the neural network.
    """
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    """
    This function uses the sine data set to train and test the neural network.
    """
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)

    # creating a matplotib graph of the the produced sine grap and actual sine graph
    # plt.plot(test_x, test_y, label='Produced Sine Graph')
    #
    # x = np.arange(0, (np.pi/2), 0.1)
    # y = np.sin(x)
    # plt.plot(x, y, label='Actual Sine Graph')
    #
    # plt.title('Sine Function')
    # plt.xlabel('Input values')
    # plt.ylabel('Output values')
    # plt.legend(loc='upper right')
    # plt.grid(True, which='both')
    #
    # plt.show()


def run_XOR():
    """
    This function uses the XOR data set to train and test the neural network.
    """
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)

    xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_labels = [[0], [1], [1], [0]]

    data = NNData(xor_features, xor_labels, 1)
    network.train(data, 10001, order=NNData.Order.RANDOM)


def main():
    """Runs program"""
    print('Iris Run\n'
          '------------------------------------------------------------------\n')
    run_iris()
    print('Sine Run\n'
          '------------------------------------------------------------------\n')
    run_sin()
    print('XOR Run\n'
          '------------------------------------------------------------------\n')
    run_XOR()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/Assignment_Five.py"
Iris Run
------------------------------------------------------------------

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.9149123511812719, 0.812299634492572, 0.8126542565314275]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [0.9141053905814639, 0.812989584285307, 0.8098444885119025]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.9132199497874331, 0.8101046552704447, 0.8104782959337331]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.9123944069130854, 0.8108298896540809, 0.8076381055741408]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [0.9114591772141961, 0.8079029948903573, 0.8082855835113677]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9091246130457216, 0.8028959586628062, 0.807040825955504]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9096906828013466, 0.8004638412088746, 0.8047357236635877]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.9106470922380261, 0.7988107780002097, 0.8029949016151773]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [0.9098312280307841, 0.7997007155948844, 0.800069527208134]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9069321611661957, 0.7937870850606497, 0.7982120031793153]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [0.9090435603387138, 0.7935080616935377, 0.7978115679647585]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.9079147191905738, 0.79007016812287, 0.7984005986056673]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9055776757062483, 0.7888814065443828, 0.7933969169427729]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [0.9072641993515973, 0.7879246448721754, 0.7923322744574258]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.9061454567529734, 0.7844056431419755, 0.7930078977098015]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [0.9053081079887754, 0.7855255347692833, 0.7899531759249382]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9032849767430463, 0.7806145127018045, 0.789427188339022]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9029530539920922, 0.7766928593808614, 0.7856676069020123]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [0.9045070355821593, 0.7752568863736342, 0.7842379758643352]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [0.9035174753645373, 0.7717347533035337, 0.7851855002699875]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9012169894619413, 0.7662880820711744, 0.7844324780236156]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [0.9026049687257302, 0.7644740443867247, 0.7827677673721728]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9006813727006215, 0.7595420924761939, 0.7826120018859107]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [0.9016715995374606, 0.7569500353267433, 0.7803213019246366]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.9005625565511611, 0.753041008853757, 0.7812476585955258]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8979577202159835, 0.7522482585460349, 0.7758657338491801]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [0.8996381836618271, 0.7504736855713917, 0.7743576416406074]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.8982866490113462, 0.7462542049923757, 0.7751560124966249]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [0.8973945969910377, 0.7478611335274398, 0.7718378035521779]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.8961188058343759, 0.7437084823047265, 0.7727709102074174]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.8938352778578347, 0.7435597610302562, 0.7677540559755113]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [0.8951731167951121, 0.7411714884246803, 0.7656324147952563]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.8939947414904851, 0.7370742476304898, 0.7667699285771985]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.8909139977529997, 0.7304601807887405, 0.765608272167476]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8902348105343395, 0.7253559382259184, 0.7609894661809806]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.891872677821375, 0.7229222593760587, 0.7589992874196497]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.893204496924002, 0.720092772159704, 0.7567063101312269]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [0.8920847152823509, 0.7218967491617073, 0.7529317368950493]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.890804073507628, 0.7174798814483793, 0.7541836107750314]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [0.8895373447002953, 0.7192208541322362, 0.7502805222807009]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [0.8882142715332737, 0.7147781837516999, 0.7515706979103929]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.8850381008981759, 0.7081452721460105, 0.7506994533380075]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.8869120112998924, 0.7056792554949347, 0.7488380730032125]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.8846400254287007, 0.7063398354622835, 0.7437370900313354]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [0.8859032376465535, 0.7031808740645082, 0.740946390307696]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [0.8845237849245117, 0.6985600476075703, 0.742365197107189]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.8830665155904659, 0.7004906627989944, 0.7382139139406282]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.8817320341769112, 0.7025285275899502, 0.734126632676005]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.8802481500375798, 0.7044442491322902, 0.7298881683368215]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.8787337418829775, 0.7063355799731593, 0.7255903101164713]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [0.877354938264394, 0.7083208053263439, 0.7213433549787308]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.8738736934662336, 0.7015953493048829, 0.7209590325763078]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.874154705291328, 0.6971310919380046, 0.716739294040457]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.8761386285720172, 0.694474386164317, 0.7142116288766824]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.8746140208279052, 0.6965375457905798, 0.709747704988738]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8720070419844723, 0.6973199171085797, 0.7040923550480643]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.8720686232127696, 0.6924815414183336, 0.6994154005051455]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.8724053949168463, 0.6878986783803186, 0.6949557297377351]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [0.873819689066926, 0.6845939889729775, 0.6914372524600861]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.8704715301311717, 0.684778512379879, 0.6850065298812492]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [0.8724627787873701, 0.682062876726517, 0.681960724856698]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.8707816827544792, 0.6771930675594826, 0.6841682512679224]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [0.8691535418039367, 0.6795217342865066, 0.6793813693198049]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [0.867444291467974, 0.6746602857649333, 0.6816473431033178]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.8652712645318659, 0.6766824322754093, 0.6765638497701074]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.8639218561910513, 0.679285793924422, 0.6719328034903101]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8601301859105573, 0.6794171793021995, 0.6652836150098768]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [0.862288722909986, 0.6767226156373066, 0.6620850629153459]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.8604736090119838, 0.6718324880685029, 0.6645974884080542]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [0.858824072498563, 0.6743712799358579, 0.6597372339195624]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.8554986550119075, 0.667978572736143, 0.6611358101134867]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.8571687846570847, 0.6645201473163496, 0.657319447140254]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [0.855305697857408, 0.6670679679386072, 0.6523355290589629]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [0.85330740223171, 0.6621054747830136, 0.6550062912477924]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.8512517665895063, 0.6570808924244225, 0.6576192961656481]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8477600131417936, 0.6582443966643275, 0.6514199940444173]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [0.8495871724550589, 0.6547514550522603, 0.6476143702856397]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [0.8475087617903391, 0.6497211201454599, 0.6503647172741317]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.8415164557276725, 0.64123737035082, 0.6499461929998617]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8444464676553849, 0.6384182107125421, 0.6470401048527692]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [0.8461872787753076, 0.6346003941625392, 0.6430929769621933]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.8438657001910544, 0.6293759230442866, 0.6458372737092574]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [0.8418276263632949, 0.6324910129554775, 0.6408314856128858]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8378349315018654, 0.6259000139895748, 0.6423826269478233]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.8398297391105218, 0.6221694415122427, 0.6385705755494357]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [0.8377253508152606, 0.6253625469343774, 0.6335192151733157]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [0.8354033612133356, 0.6202087870673185, 0.6364805444035581]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.8320670784350098, 0.6142065471754347, 0.6386780435698638]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8305970058199116, 0.6076932259826119, 0.6322121377267563]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8316767584354993, 0.60311216224276, 0.6276668106379137]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.83133211459586, 0.5973470292939321, 0.622158052373733]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8330247334529479, 0.5932371631046193, 0.6178521662188038]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.8353572941228885, 0.5893883003080983, 0.6139511230224431]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.831648692971112, 0.5832891604043832, 0.616274475440765]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.8333037641865393, 0.5790031963413418, 0.6120019749160494]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.8310865298225125, 0.5828092553401357, 0.6068773925615722]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [0.82865189616716, 0.5865210021262505, 0.6016962061558746]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [0.8261212225689898, 0.5813328797382878, 0.6051203524530181]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.8235122000334512, 0.5761476662867911, 0.6084867836807573]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.8192273001233191, 0.5787965487987348, 0.6022635284432756]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [0.8214950805654635, 0.5748228809564302, 0.5981606413179892]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.8166700483609239, 0.56834547155535, 0.6003706309868395]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.8192107074668457, 0.5645250080028189, 0.596459925250225]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.8166945938270301, 0.5685119511113375, 0.5913102027448245]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [0.8139848358485926, 0.5724214186116596, 0.5861337656307642]

Epoch 0 root-mean-square error: 6.592215173136301

Epoch 100 root-mean-square error: 3.463263730497193

Epoch 200 root-mean-square error: 3.2326234495261224

Epoch 300 root-mean-square error: 3.137183804693217

Epoch 400 root-mean-square error: 3.0884981255607076

Epoch 500 root-mean-square error: 3.0537754747594823

Epoch 600 root-mean-square error: 3.0397977644436627

Epoch 700 root-mean-square error: 3.0192118456885244

Epoch 800 root-mean-square error: 3.0101089980564053

Epoch 900 root-mean-square error: 3.0128038529966834

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [0.00020439156423843996, 0.36665200705660234, 0.8666821972261094]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [0.0001880778595685787, 0.3645520831792812, 0.8764174456120647]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9270015261634157, 0.26429990199498604, 8.658610585628889e-05]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9262703954528494, 0.2633506253592296, 8.744176565666481e-05]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.00027590131161915984, 0.35570686085783026, 0.8280491944727577]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [0.0004865965569597633, 0.3477568428774302, 0.7305732227189375]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9254558257412983, 0.25936271116072773, 8.924195013567312e-05]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [0.00025009052445107037, 0.35022804516735834, 0.843351718712482]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.003454778441183974, 0.3241469561770547, 0.2709129104017575]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [0.000348880099200202, 0.34904924683882066, 0.7925184665914646]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9274441280485415, 0.2579933453958443, 8.674780306596624e-05]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9264220928422826, 0.2571795352889844, 8.81664409952256e-05]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9262868067041985, 0.25634495546852515, 8.866943282967397e-05]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9279077800027413, 0.25512399489691423, 8.620963603653445e-05]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [0.0016132054373635138, 0.32836447058592544, 0.44688364955003096]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.5803622624974974, 0.2746967032133613, 0.0008242374935353457]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9267756516675499, 0.25979959407970804, 8.548115041484035e-05]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [0.0002684671065958516, 0.35139424395729885, 0.8297806606167046]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [0.00023149162307327327, 0.3501050425777939, 0.850411611052736]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.025689124555812008, 0.3057258804883002, 0.04400127562191323]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.924885798846328, 0.25887688530864783, 8.850330596145703e-05]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [0.00022440728305812233, 0.35085012121294296, 0.85478394698248]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.00021343149541239992, 0.3486545664802285, 0.8613196581878291]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9181156251604121, 0.25570602502520157, 9.812256082394996e-05]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [0.0001804002686887508, 0.3464372440845487, 0.8808421077799998]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.038031410784817185, 0.2975234307483632, 0.029754298546271422]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [0.00021994931544457318, 0.3461720453513048, 0.8581733000678843]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [0.00017748784900663931, 0.34549693787272306, 0.882998297156667]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.11855140420123078, 0.28701223583514995, 0.00877058157496239]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9265362740948011, 0.2548505302896297, 8.691232071995885e-05]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.02108428979477657, 0.3042993299474357, 0.05398673382335927]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9268267061970609, 0.2566882466056748, 8.649725115494909e-05]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.005439236844993031, 0.3185869217101004, 0.18726951569955516]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.926148408768356, 0.25865869400006536, 8.716196978009562e-05]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.005761070615187662, 0.3209626677968866, 0.17802011739270224]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.004514706727162783, 0.3271235949132664, 0.21694904388005448]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [0.0001853239323340704, 0.36089607917112826, 0.8772112588984335]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.030257660151185563, 0.31162657347657685, 0.03734430373967392]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.926229949834024, 0.2646969879204817, 8.676554461001613e-05]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9274459049924881, 0.26352436212051816, 8.513453237421036e-05]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [0.0002668156551098431, 0.3565442468712404, 0.8317324980032181]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.043679243875431865, 0.3075330761628198, 0.025736486306581]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.08253434083442422, 0.30514066093669945, 0.01307517323348753]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.005401472240411524, 0.33385493152596885, 0.1874510117882552]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [0.0006474693483367517, 0.35805896270067156, 0.666795669561147]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9274979613430256, 0.26794467876337813, 8.557888779227219e-05]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [0.0002192843829297392, 0.36459945727240084, 0.8590210186998015]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [0.00017415765559677368, 0.3639333283569796, 0.8853737859682025]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9269034290094983, 0.26371896363409364, 8.669974625932006e-05]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [0.00017298742394204923, 0.35993756769753743, 0.886296839851961]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9266683521391441, 0.26117033061849254, 8.720787236905672e-05]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [0.00019203300617061072, 0.35499140211475116, 0.8753643286368332]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.015052074773125127, 0.3133885131154473, 0.07546216104450987]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.015849903502999266, 0.31667112188698265, 0.07177241464999078]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [0.0002728727411395445, 0.35767253040204705, 0.8308985192613109]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [0.0003647232969615806, 0.35226859733967897, 0.7858129857352674]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9266222990537238, 0.26119266248846146, 8.78198441385759e-05]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.007995235708071305, 0.32087795987956735, 0.13619098988203424]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.925196529932164, 0.26325188449836373, 8.960919088970845e-05]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9245046887023195, 0.26238422088558366, 9.070547922398487e-05]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [0.0002869687366228497, 0.3527998606044249, 0.8247515602755076]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [0.0002600687683848431, 0.3586041983822761, 0.8364372197558415]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9270708820323195, 0.2623455232851192, 8.642216636486606e-05]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.03033008812459807, 0.31132691962854386, 0.03765263238782301]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.002302202767718163, 0.3385452297280657, 0.3573734245939716]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.018656625201000553, 0.32330453363263056, 0.06040844897956387]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [0.00019164116192732684, 0.37092278457276534, 0.8737536698288285]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [0.001279682517500329, 0.34975134632405663, 0.5006706483480065]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.002297937932447886, 0.3418914240951837, 0.3594843056517833]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [0.0001481029470812158, 0.3726049884494286, 0.9007041913738741]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9255478231529128, 0.26816864669312446, 8.834168196227008e-05]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9263128167433924, 0.2670482189051449, 8.738079449328314e-05]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0018492968457735788, 0.34315005518301095, 0.40984956301823516]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [0.0016298172599073506, 0.3486983004112932, 0.43809307098448436]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.004460916455487326, 0.33687818422111016, 0.2211462835803816]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9257604903584233, 0.270360317813171, 8.816514927417904e-05]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [0.00018116596653627715, 0.37077320078159204, 0.8811242844085737]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9261079504887887, 0.2675760508190296, 8.787733182905066e-05]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [0.0001942885106275607, 0.3659022392950961, 0.8737210286505056]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9267309654595599, 0.26480123999843225, 8.715527294081971e-05]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [0.0001810691922114975, 0.36249964882038527, 0.8817125821096585]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9275182050276104, 0.2620586026147435, 8.618545934662224e-05]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [0.0002125990548752192, 0.3569607480309788, 0.8638244257641039]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [0.00019387171423769454, 0.35506910224875393, 0.8747559316430634]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0069270438437218345, 0.3201120284107622, 0.15420258567264583]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.025423067528668512, 0.31233837024662436, 0.04527354165151939]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [0.00041821287455389263, 0.35382776128557575, 0.7612429021318733]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00045379148696590336, 0.3504714366040284, 0.7469840611906825]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.13998050693619618, 0.3015705115476319, 0.007207094151273677]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.002875456462478826, 0.34161889583003274, 0.3054784776120679]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [0.0030566105298516105, 0.3452768424830057, 0.2906903816648739]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9274147775509728, 0.2736690295098935, 8.438215348824195e-05]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9279090879372628, 0.2725174334217513, 8.374790589015545e-05]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [0.0008993522124957674, 0.3590022083896717, 0.5862554421744712]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [0.00016571361094335017, 0.3730219650592703, 0.889400178664279]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.13509606410680222, 0.305897744158175, 0.007498161350603711]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.022717659837873324, 0.3266790196886322, 0.04983461431232298]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9260648311434668, 0.2742356947693782, 8.720850968909232e-05]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.926379653660837, 0.2730472656940501, 8.659843157544846e-05]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9262592541063055, 0.27215344950763326, 8.737272681651576e-05]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [0.0002593007781572274, 0.3702396865034807, 0.8361166175151263]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [0.00024823569657121664, 0.3677594373159726, 0.8425519931775545]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9260411689506828, 0.26767608989436287, 8.785996767347423e-05]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.06576779471438833, 0.31097094143494375, 0.01681384520878136]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9252477665292764, 0.269613099465294, 8.879444497205901e-05]

Epoch 1000 root-mean-square error: 3.0025897907649552

Epoch 1100 root-mean-square error: 2.9593995933282335

Epoch 1200 root-mean-square error: 2.9439419493677823

Epoch 1300 root-mean-square error: 2.9874003039730064

Epoch 1400 root-mean-square error: 2.9633785487376176

Epoch 1500 root-mean-square error: 2.996480986861723

Epoch 1600 root-mean-square error: 3.0046479775908854

Epoch 1700 root-mean-square error: 2.950149082139943

Epoch 1800 root-mean-square error: 3.009624103274965

Epoch 1900 root-mean-square error: 2.963347070495349

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.059806719170795476, 0.30397164515063907, 0.002274170188080175]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [0.00013705028307449152, 0.3698489856067617, 0.6750070058340741]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.005436336223763534, 0.32923592681101077, 0.03385832940243888]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [3.4993515247620566e-05, 0.38608396154278535, 0.9052122546591129]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.0016064086546631693, 0.342841971354515, 0.11999869623866688]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [4.199499824063313e-05, 0.3856128686640628, 0.8863033772370953]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.011258584642360901, 0.32451677171103777, 0.015290246852800855]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [4.623786897054815e-05, 0.3858394054604195, 0.8752629117594543]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.0033288431574555162, 0.33803774946268506, 0.05724434251653154]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [5.987317745148568e-05, 0.3844787320768871, 0.8406451251406764]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.018388963393712335, 0.32858614378591544, 0.00874437585208312]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [3.742630659364846e-05, 0.3991385166902491, 0.8973464408058524]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [5.07670956116066e-05, 0.3923693265804562, 0.8618940841807528]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [4.220693705358463e-05, 0.3911102248850565, 0.8847732422228072]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [4.688064883220077e-05, 0.38670213371393924, 0.8725488724933251]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9497061394155116, 0.25807278825914104, 4.132458439880381e-06]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9506134253141106, 0.2569162169187676, 4.035817987115786e-06]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9513993083901675, 0.25581262122887616, 3.9565858201220475e-06]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [3.886684911885392e-05, 0.38202445906010424, 0.8942172680068473]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0010593795217791412, 0.3447277169403518, 0.17735591482343446]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.006834325451152902, 0.3300701539832154, 0.02625396076054423]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9499475443861262, 0.25912414743550505, 4.096226315396383e-06]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [3.749120201853233e-05, 0.3873828391000939, 0.8977696346359139]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [9.02131725449641e-05, 0.3748050653250577, 0.768400255474133]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0034054800898655306, 0.33467199407945186, 0.05574121818066308]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [4.526408029879131e-05, 0.3837302821336351, 0.8776682961396741]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [4.173151227450093e-05, 0.3814283113941711, 0.887222618038649]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9514741205787637, 0.25382898093697737, 3.977379399135115e-06]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00022057350738535896, 0.3599091139595607, 0.5537742684720224]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.950307983597771, 0.25592506005259413, 4.032699244853135e-06]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [6.139713013569186e-05, 0.3768549916420422, 0.8343752750918206]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [4.825856498237084e-05, 0.3763593980523508, 0.8684687664482381]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9512108365327995, 0.2514838923919832, 3.962102926362566e-06]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.007503542226222626, 0.321501448155136, 0.023681279396813918]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [6.152120319355513e-05, 0.3741255835660209, 0.8348429848051553]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [0.00016668372094061124, 0.3608838604123292, 0.6264766057785346]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9507317310863128, 0.25028097050431447, 4.059126215120645e-06]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0020613095371986895, 0.3320846148752735, 0.0940433378817201]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0024630452028241785, 0.33439427274355593, 0.07839841871021225]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.005184913060044701, 0.3309976264898106, 0.03576829376992884]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [3.3512987312125245e-05, 0.38748665759560835, 0.9096042105029277]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [5.283742089267935e-05, 0.3794117613117924, 0.8587290637847216]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [5.9396197868512474e-05, 0.3751035156622965, 0.842502644690654]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9516887233013377, 0.252498406045931, 3.974923242864958e-06]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.007321136139084191, 0.32274507047455475, 0.024747041560293278]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9512191887083649, 0.2544876215436715, 4.021093526653161e-06]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0020391739944139575, 0.3381820891558533, 0.09548259233773229]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [0.00012222592988282358, 0.3713307498690521, 0.7063916387762238]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9510038710299744, 0.2592968759924951, 3.9929031196747874e-06]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [4.718200896509552e-05, 0.38518573716542476, 0.8714972723471438]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [6.729793796396693e-05, 0.37823355970151423, 0.8208489367382987]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.039086536378947635, 0.31071750115399854, 0.0037449291856438637]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.011469981196618172, 0.32653261923276933, 0.014901843053866225]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.04728423417061705, 0.3158937329204058, 0.00300427585397346]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9505595426793424, 0.2634518153687378, 4.039667990341473e-06]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9507205159294313, 0.2624652487385678, 4.0324337290445094e-06]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [3.912960608307814e-05, 0.3915654520151992, 0.8935827531278449]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [4.840800099280575e-05, 0.3859810917144026, 0.8691096831138027]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9500435273952476, 0.2580438565408076, 4.106350299311496e-06]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [4.053935689096241e-05, 0.38351515109259526, 0.8901766011557599]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9515861390536494, 0.2550661665875283, 3.958747051852782e-06]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [0.0001382444313734224, 0.36640935758464377, 0.6753100930545084]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [0.00013940224453870613, 0.36356767976359333, 0.6754533807984542]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9521883938510766, 0.25520613087161803, 3.8700995362784574e-06]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [4.393789566376473e-05, 0.37931573766024423, 0.880186912062874]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [6.77849471113616e-05, 0.3716990013616941, 0.8197660721798606]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [0.00016162568231967058, 0.35988298107889133, 0.635033401524539]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9441812713574381, 0.2508503058176416, 4.722160251171619e-06]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9508104551605933, 0.2488814537284154, 4.075716990225205e-06]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.977589570439352e-05, 0.3693046633294364, 0.8931340791757455]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00043674780594883157, 0.34245850898643637, 0.36906442417294494]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9521677498155562, 0.24883082831298747, 3.895177911866244e-06]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9506837988303584, 0.24823124748189074, 4.038054611732343e-06]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [5.8557880380042144e-05, 0.3650333021187557, 0.8434121912705727]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.002565028438060691, 0.3249200853153811, 0.07498827286049292]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.950652916040168, 0.24859831520298348, 4.0564225163759104e-06]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.2035659834443441, 0.28486165236032135, 0.0004908704712142274]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9508529765967509, 0.25038226367652616, 4.020575315055165e-06]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0001387572706147146, 0.3595803943753148, 0.6736148242022608]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [0.00015613621070469707, 0.3631132543234668, 0.6394126948282758]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [5.6712848411053205e-05, 0.370869131038586, 0.846658365399291]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9516404639975922, 0.24891439313492522, 3.925287100841396e-06]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9504167094738286, 0.24828396702303182, 4.046416963916335e-06]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.014081293729009703, 0.31119341575330894, 0.011833126196732928]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9513365075825226, 0.25000646709718666, 3.957434000967786e-06]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.031034896704829666, 0.3061497888529862, 0.004863437227073919]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9508457960169561, 0.25200027880425535, 4.007463723945855e-06]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9517027198591075, 0.2509249023714186, 3.921319636164321e-06]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9513099722630728, 0.25012343277948895, 3.961688824459624e-06]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [5.4213011206717355e-05, 0.3704139510367769, 0.8535706255466072]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.0012037766292248447, 0.33607946046036213, 0.1574700595076613]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0008674157348206121, 0.3435793977821043, 0.2115057932646676]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9506489428043626, 0.25332205085735077, 4.011542204598084e-06]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9522062249637645, 0.2521043886816145, 3.859937787025633e-06]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9520062317597785, 0.25125660597026134, 3.881358189920499e-06]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9517480340096874, 0.2504304956247384, 3.9091554404171365e-06]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [6.251803750732553e-05, 0.37091738869678237, 0.8321206187321766]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9518630985269602, 0.24790223585020657, 3.906246603750252e-06]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [8.68064733639532e-05, 0.3634970467686706, 0.7755632352572921]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.002190966323692132, 0.3281812999016425, 0.08781543092726882]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0009595644630002376, 0.3405314564827345, 0.1940966518763311]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.004675245791274186, 0.3286834265987099, 0.03955380065387476]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [4.824108304136068e-05, 0.38064971471795367, 0.8690431987253796]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9510113710921752, 0.25229413738163353, 3.999489651282608e-06]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9515369846475723, 0.25127286400626725, 3.9442716995398975e-06]

Epoch 2000 root-mean-square error: 2.964797047713175

Epoch 2100 root-mean-square error: 2.9347972753004634

Epoch 2200 root-mean-square error: 2.9363570953235576

Epoch 2300 root-mean-square error: 2.951403887609159

Epoch 2400 root-mean-square error: 2.923891202124146

Epoch 2500 root-mean-square error: 2.938583728642433

Epoch 2600 root-mean-square error: 2.9642107469200867

Epoch 2700 root-mean-square error: 2.9538073064786703

Epoch 2800 root-mean-square error: 2.989562938740425

Epoch 2900 root-mean-square error: 2.9683009340602533

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [2.666342613288269e-05, 0.38900247804438837, 0.8480487238766745]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0038577128089895777, 0.3304222637519318, 0.014286135263718445]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00024450210890804796, 0.36494531295893234, 0.28302197699433285]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [8.171232356475369e-05, 0.38212563232848085, 0.5924608873015307]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.832342868372043e-05, 0.3965283495303158, 0.8981479511911842]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9613430181267347, 0.2459945541514868, 4.0537104705068727e-07]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [2.2041972300334844e-05, 0.3899294427080799, 0.8762363549681077]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9606626899995417, 0.24367853305736642, 4.1701359560483416e-07]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [1.877848854306149e-05, 0.3874509709886306, 0.895785285703062]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [1.9326490011338107e-05, 0.38388366335756147, 0.8926585984188782]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9610657995114283, 0.23935027425831965, 4.109535854575591e-07]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9616662347978853, 0.2384072058364999, 4.0285097545570375e-07]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0022417078587982564, 0.32699638242844725, 0.02735383410130682]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [2.0474828487679178e-05, 0.3826171244863274, 0.8860397280954077]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9605166829215118, 0.23895975939842645, 4.1904357084902475e-07]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9596723102205712, 0.23835961178150925, 4.3022683345085865e-07]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.021373170139060413, 0.30304733279819357, 0.001849482779394703]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9609421878911353, 0.2399537790327937, 4.1322002344722436e-07]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9601815223001474, 0.23934490577330525, 4.2364936347178754e-07]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9595449676770468, 0.23869958773930694, 4.3222863959740994e-07]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9619143715432038, 0.23734068665577795, 4.003242648964424e-07]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.953884919146343, 0.23837351945305824, 5.107881298656967e-07]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9594495862963586, 0.2363576085214394, 4.331995218722107e-07]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9603697680779688, 0.23537960032081923, 4.209503003099605e-07]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9619879353989235, 0.234242773256817, 3.996554900476713e-07]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [2.0930595459529972e-05, 0.37339163510082996, 0.8837101199222623]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.8588311119890476e-05, 0.3716917939921335, 0.8976922163223]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.398824983461134e-05, 0.36206176258890943, 0.8101514342092536]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [1.8450201488962804e-05, 0.3659132247276218, 0.8990253795615178]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.737442083864527e-05, 0.3636372842327774, 0.9054859580941732]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.00010378735529649231, 0.3418234520927272, 0.5303623926518032]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9609395300112107, 0.22882527274222852, 4.1212405386978187e-07]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [2.5138507007335494e-05, 0.3606959458531005, 0.8581791083775769]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9613774371132886, 0.22650918234800174, 4.059952433170329e-07]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9617147946473689, 0.22574445092661594, 4.0172402067200153e-07]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.008038793380040828, 0.2969788690706753, 0.006009766703817805]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [3.463609183688221e-05, 0.35690507149809336, 0.8052645917822587]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [2.5567575275599812e-05, 0.35743719792525214, 0.8565882102130161]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9619473768190355, 0.22470226098695398, 4.006254921188635e-07]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.842827412590943e-05, 0.35726327850260037, 0.8986109942698768]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [1.9569454352469515e-05, 0.3538316406233935, 0.8920001351846691]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [2.513874612032727e-05, 0.34848459655420916, 0.8597462957278691]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0015026657671029227, 0.30498942812713764, 0.04384410057731904]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9601803319692188, 0.22281023717017853, 4.2685814183903683e-07]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.011927415712056511, 0.28805127800437286, 0.0037804059545544755]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9615712112928033, 0.22433510528539224, 4.073220144876378e-07]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [2.1882352104364587e-05, 0.35417782731570585, 0.8788681250774232]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [1.8870821924151943e-05, 0.3529967848751776, 0.8966804939038139]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9600302992370028, 0.2212077825387924, 4.293824744231009e-07]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9614775419965212, 0.22022420469408135, 4.0950686442328653e-07]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0009843844101891839, 0.3087980958367159, 0.07102190724843427]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9615849239405017, 0.2220907434237258, 4.079113778804216e-07]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [2.3636344705069917e-05, 0.3498440047157742, 0.869062994741859]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.0008816955049287474, 0.31069866359195153, 0.08029917505377651]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9601471353589222, 0.22293140986752372, 4.279852903661573e-07]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.004098810733768615, 0.29878723941546664, 0.01362694578059563]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00034319488054377955, 0.327296779982855, 0.21272051913571471]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00044552057243504357, 0.32886698985143975, 0.16439270284728047]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9611215870350139, 0.22980531618768785, 4.124581727115544e-07]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [5.379934038610847e-05, 0.3546940822134114, 0.7113473340686048]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0030825242370950885, 0.31001557879652414, 0.01909532381231138]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [2.091938515885857e-05, 0.3667167101735505, 0.8850467525310226]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.984168140219875e-05, 0.3643692914312582, 0.8914945669078107]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [2.550756853170631e-05, 0.35877497655271556, 0.8589936352381533]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9612898806923349, 0.22577140700648698, 4.140405984032396e-07]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [2.116641722839008e-05, 0.3570772214496755, 0.8842163063791683]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00016822614374364955, 0.3327550963338299, 0.3900910274148459]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [2.0911966198778693e-05, 0.3592144520140369, 0.8845407091647849]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9616862215013005, 0.2247169422022744, 4.0622083655687997e-07]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [4.58135364988568e-05, 0.3472406825858881, 0.7501330254231657]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0404803805682951, 0.2831183342159782, 0.0008348541602742621]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.2453681882443917, 0.26718290550294305, 7.254469592330381e-05]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [1.7370448120082333e-05, 0.37035012741400264, 0.9031887246441133]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.861633365004359e-05, 0.36659740379905054, 0.8958123851701495]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.001976603000003509, 0.31505114366556836, 0.03133000108902323]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [9.954822510431964e-05, 0.35013042819940626, 0.5366040056491476]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [2.4190134052381477e-05, 0.37024793702721687, 0.860632111223731]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.09819191904471801, 0.28103010237078846, 0.00026395408889327154]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.024296707022606733, 0.2985599460450419, 0.0015379439326461415]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.006554246822671833, 0.3154574643412851, 0.007486967889512143]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [2.2067200491467077e-05, 0.38101137214325875, 0.873417418734581]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9616109708637862, 0.23867132114837333, 3.9544337698806697e-07]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [3.2838746053906465e-05, 0.3724546342153634, 0.8112704427247567]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9617094171765506, 0.23628098085630164, 3.9522189618826584e-07]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9609363005691112, 0.23569575564413497, 4.053699792326238e-07]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [3.3327396367200774e-05, 0.36737065571056954, 0.8092421617964745]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9602576923276501, 0.23354796646261589, 4.1537236006019035e-07]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00039248879250939554, 0.33722868932905364, 0.1822306909387428]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.029808126587487865, 0.29680601788108163, 0.0012041861774441031]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.07447824230230041, 0.2906917873188645, 0.00038081774990375767]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [4.68044964631543e-05, 0.37268326450789047, 0.7385904239142573]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [5.510052107964862e-05, 0.3679819518168841, 0.7006924500377146]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.008916913907016327, 0.3114496653607387, 0.005281963789658876]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0036540143987267154, 0.3243774353358629, 0.015302970908769685]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9596428434841989, 0.24318343855630375, 4.2713315979595105e-07]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.0016615687086212335, 0.3356344589890628, 0.038473862399175635]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [4.1752833165997534e-05, 0.38053174976712945, 0.7669982036285186]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.5678680410279674e-05, 0.38857761222858395, 0.9144832543503356]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [5.291707037344055e-05, 0.3717009007048506, 0.7139693478377483]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9609244466456789, 0.2441502807064814, 4.06475666589966e-07]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0010777146477505247, 0.3422555352832359, 0.0622407521782238]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9609124373120181, 0.24606214592342615, 4.0655471309828684e-07]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9607490815120202, 0.24526708769318284, 4.090108695084553e-07]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.02470319903724347, 0.31077747914570514, 0.0015182980381977307]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [7.268892652858522e-05, 0.37858253967440103, 0.6258076166552121]

Epoch 3000 root-mean-square error: 2.93208284050679

Epoch 3100 root-mean-square error: 2.944573198462866

Epoch 3200 root-mean-square error: 2.954956841467432

Epoch 3300 root-mean-square error: 2.934198972376173

Epoch 3400 root-mean-square error: 2.9320614232327618

Epoch 3500 root-mean-square error: 2.932565244171043

Epoch 3600 root-mean-square error: 2.8849398305943175

Epoch 3700 root-mean-square error: 2.923320580771232

Epoch 3800 root-mean-square error: 2.9191096182523557

Epoch 3900 root-mean-square error: 2.934724411170819

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9704606634787879, 0.22937125032981476, 5.675570202790982e-08]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9701240817219051, 0.22876686214607966, 5.7623657989421773e-08]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.00352703842108645, 0.32353165323673655, 0.006778455379807304]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0004649623397445657, 0.35154561428159053, 0.0840073940264454]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.4663311655833019e-05, 0.3992852506538538, 0.8845674039665256]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.00074347528930442, 0.34719852988856276, 0.047898788439736516]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [1.3973179318817073e-05, 0.40130168575066083, 0.8908636976081257]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.4538382049420012e-05, 0.3973628816219125, 0.8859916225501125]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9689623905886793, 0.23120821919374207, 6.082650394650502e-08]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9702808439151891, 0.23004318630645132, 5.737838854471185e-08]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2415581583978017e-05, 0.3940289469459247, 0.9050339557561387]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.2494309964903987e-05, 0.3906019161439932, 0.9044412988896019]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00022048261785276047, 0.35235743655620855, 0.19351647790019424]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [2.5404143200562955e-05, 0.383342309648796, 0.7918153851125862]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [1.8852815955614217e-05, 0.38393829059033835, 0.8484948690493473]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9695339291316794, 0.2298602466093386, 5.884570065905491e-08]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [1.4534552204977445e-05, 0.3913019519097029, 0.884929060246793]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00014131869450706945, 0.36004849608866035, 0.2951814117583665]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [2.1484370054603174e-05, 0.38803067791790236, 0.8224154499779172]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9693412359537116, 0.22851576107606092, 5.913509356427072e-08]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.6845246169650604e-05, 0.3869282445344038, 0.8639045586230472]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.00668056777849641, 0.3127461625844473, 0.0029665430113048345]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [3.671725495022775e-05, 0.37850353779652546, 0.7011769542668627]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9690154579887644, 0.22733894596759818, 6.051177914261963e-08]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9689057620555102, 0.22667120386403036, 6.079571254640823e-08]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9689055446154237, 0.22597923952303234, 6.081596094101015e-08]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [1.5753641296303038e-05, 0.3831525230913487, 0.8749794786470774]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00025008005776109226, 0.34671377078200366, 0.1690032954646335]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9695696920228983, 0.22609577144243717, 5.9078310068173315e-08]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.00429196775028657, 0.31704950386464614, 0.005268764263620399]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.969660446246193, 0.22796213588703249, 5.881706238864374e-08]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.3088060662165774e-05, 0.37823518645353216, 0.729963179191765]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9682232193371105, 0.22612236669761895, 6.280726606407716e-08]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.668542159064687e-05, 0.3827783412232902, 0.8675144472779356]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9693095061859379, 0.22356196018082372, 6.022933027502753e-08]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9688161333445142, 0.22304533006605307, 6.154506117559133e-08]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [1.3600746605522135e-05, 0.3803167484769416, 0.8950258640221004]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9700650794471004, 0.22041622511646392, 5.8232923294852183e-08]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0011950050096368602, 0.32353754719618344, 0.026897692600994852]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [1.1758075235063144e-05, 0.3827493731844359, 0.9114178196366048]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [1.2826919795728534e-05, 0.3784944884537461, 0.9020934927249782]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.002952245921537629, 0.3121402881970899, 0.008601924028360916]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00015678757941816742, 0.34981129924267024, 0.2720907450108863]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9706062989222861, 0.22414897306922044, 5.6589385091614784e-08]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.969590902885349, 0.22379299482884038, 5.919783002350394e-08]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0005284087550305808, 0.33816319594596267, 0.07259328613415016]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9693744854268261, 0.22577316963287233, 5.97575303817529e-08]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.5541951570607493e-05, 0.38446118411524005, 0.8774244724194428]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.004006094953758105, 0.31533504700976744, 0.005796573414390369]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.968922477058679, 0.22623770165199716, 6.109795489502348e-08]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.8574173507581402e-05, 0.38265124002099976, 0.8509465775022638]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [1.2592858237111294e-05, 0.3843346910299819, 0.904006441539248]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [2.2648747164353568e-05, 0.37394009423449537, 0.8163824345980465]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9706302377519416, 0.2203268444503441, 5.690971841852489e-08]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0014307517163109488, 0.32151648818751166, 0.021556261180691004]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.2378024049029535e-05, 0.3820676666510503, 0.9063545264299729]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9701577997655052, 0.22082840516297153, 5.821455758381745e-08]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9701626246755846, 0.22016942844268486, 5.818720709907529e-08]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.002052112386874431, 0.3171243498371405, 0.013694253780843194]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.00035670217719266237, 0.34127258664804827, 0.11576909473762921]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9638471708575117, 0.2265167989460971, 7.523748692631734e-08]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00036343520688714956, 0.3444579045405646, 0.1131357486866948]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00012682867169018503, 0.36156242941173, 0.32915573830892353]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9698901787844464, 0.22922431443233998, 5.8381428529248456e-08]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9703525467410676, 0.22835474661008814, 5.718610267846455e-08]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [6.188370991505437e-05, 0.3732116233199619, 0.5491818028855403]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.00025267722609168653, 0.35312432586708326, 0.16948020662870983]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.279913156185568e-05, 0.39499538829394254, 0.9027423748105431]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.523424641109029e-05, 0.3894330352464138, 0.8814536010354171]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9698644724938086, 0.22558696510732026, 5.908544253031602e-08]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9695575999990681, 0.225051086713684, 6.011752569046696e-08]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.1365242273793243e-05, 0.38799117143601936, 0.9155681182634574]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9701186149299165, 0.22252684421527738, 5.843304439195452e-08]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [1.3251706794091345e-05, 0.3819014974971568, 0.8991718887065232]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00010942728443746079, 0.35314455224122165, 0.374086316558969]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.024996354943444925, 0.29469546166360944, 0.0005448041581556463]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0002722606973626131, 0.3507395241314109, 0.1552122910755561]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9703532947163303, 0.22793241982796988, 5.728879395468851e-08]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [1.9105913562728116e-05, 0.3872697891814122, 0.8462620367395818]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.345911313256013e-05, 0.3884845656058792, 0.8963373130095005]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.969933895773018, 0.2241547331863677, 5.8592319567415326e-08]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9704816806634654, 0.22328931158633553, 5.713230423148587e-08]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9701577940562731, 0.22272105897627725, 5.796389659383912e-08]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.7870959570875703e-05, 0.37895803278556894, 0.857647131216986]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [1.4885148692969977e-05, 0.37812975797777704, 0.8841703895884416]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00016469194373693873, 0.3461589617457171, 0.26029038144170175]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.970319002140341, 0.2214947803481863, 5.747953678207812e-08]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [1.5704749425694995e-05, 0.37835193110706994, 0.8763957043552593]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.260999922875836, 0.2624581463965228, 1.8967720788355583e-05]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [2.2143351883925597e-05, 0.37481932021173975, 0.8195489629855597]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.013292561784541844, 0.297667624046486, 0.001237742435925138]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.4508295003608023e-05, 0.3813354255378031, 0.8868206360971158]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [6.277223921463046e-05, 0.36043173522472616, 0.5459343874207477]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9702770763993607, 0.22373914400659964, 5.670890000424627e-08]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.02474609536955455, 0.29483659244687305, 0.0005424135625686978]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.036044609269000015, 0.29398387322884867, 0.00033017743129228007]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [2.4594270921973287e-05, 0.38417236639170893, 0.7965731198626307]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0017072639982573085, 0.3299372261782509, 0.016931975308046732]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0296389308271444, 0.30085532122923236, 0.0004293844139081588]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [7.302711055256898e-05, 0.37623525212439424, 0.4940149361026141]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [0.000657612648090366, 0.3535845533255825, 0.0544156074391578]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [8.236332814395518e-05, 0.3840525891163757, 0.451294552517147]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.612585077245853e-05, 0.40198719501628316, 0.8711792740867236]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.7753135698269716e-05, 0.3973344365536126, 0.8569834924268697]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.968987890398832, 0.2324337761582176, 6.04543776820874e-08]

Epoch 4000 root-mean-square error: 2.899921381434991

Epoch 4100 root-mean-square error: 2.9292781524475364

Epoch 4200 root-mean-square error: 2.9284650311680407

Epoch 4300 root-mean-square error: 2.911078789468561

Epoch 4400 root-mean-square error: 2.9588131111561897

Epoch 4500 root-mean-square error: 2.928804846338833

Epoch 4600 root-mean-square error: 2.9406695038800534

Epoch 4700 root-mean-square error: 2.8936864242776137

Epoch 4800 root-mean-square error: 2.898417822419437

Epoch 4900 root-mean-square error: 2.939983927248893

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9750778382614637, 0.22244543635268282, 1.0506342252919862e-08]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9750851993547329, 0.22175231384053692, 1.0481518474078168e-08]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [1.155181645256533e-05, 0.3968682075774312, 0.8922728799631657]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.2361960733020392e-05, 0.3925928242015913, 0.8832232953059008]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9757979798410851, 0.21756580140787957, 1.0080866196102603e-08]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.498335564678939e-05, 0.38588522127595304, 0.8536021098625165]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [2.0971750536154472e-05, 0.3783419627318244, 0.7871017598672314]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0032691612958880996, 0.3130037452667825, 0.0037836514409916766]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [3.5425414716798704e-05, 0.37302818184980113, 0.6452503116304984]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9762178100885047, 0.21871483245578655, 9.750954154314622e-09]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.11266070164752158, 0.2776120537464205, 2.5582551556624703e-05]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.001228314737825181, 0.33565072330445217, 0.01396367801268817]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [1.7460165804544572e-05, 0.3950224996250782, 0.823388778462556]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.005773063178031774, 0.3178918971988262, 0.0017112194475967304]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [6.418338976473129e-05, 0.37895024854167386, 0.4425994882201963]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9759169408694631, 0.22672236196694381, 9.833723253997573e-09]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9758346634478781, 0.2260766411892775, 9.893169921748659e-09]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9758295180789944, 0.22537215561660454, 9.884370523303562e-09]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9762000036319501, 0.22452318823669828, 9.66840071308888e-09]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9759017402091809, 0.22401462550413567, 9.874013086085492e-09]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [1.655535716969968e-05, 0.3972245692260126, 0.8321017402544636]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9758343460638861, 0.22168942565017977, 9.911787157368557e-09]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [6.73075986795564e-05, 0.3743805923722516, 0.4239308894525782]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.6098328818750474, 0.2570469318285717, 8.209333551319797e-07]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.2051740665734185, 0.2797373573223015, 9.345767727966843e-06]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0001299706092851682, 0.3776335057072018, 0.2237540781118639]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.624966716479253e-05, 0.4106860284084778, 0.8295258482859861]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9748329751924447, 0.2296486899152998, 1.011336705387209e-08]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.926958710177132e-05, 0.40385397372695053, 0.7947429785772065]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9754362705307836, 0.22696405408060602, 9.80567108775088e-09]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0008265723296921648, 0.34975440181169687, 0.022721354798023074]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9759951166871779, 0.2286194983441964, 9.487434916272445e-09]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [1.2748353521189143e-05, 0.408950252992402, 0.8722956886189895]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [1.6091667397428594e-05, 0.40224861351551733, 0.832893064555582]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9707776490952881, 0.22652379690792984, 1.2590164334845983e-08]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.012196461365510365, 0.3126004994603928, 0.0005882035224023755]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9757636872507649, 0.22647471208505762, 9.655879825753675e-09]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9756059831279326, 0.22584767048303375, 9.745052137165824e-09]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9739300341046845, 0.22583210610515367, 1.0694262578394698e-08]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2777964795757728e-05, 0.40262116109052676, 0.8725900083806944]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.974281924454535, 0.2233174756640526, 1.0508705849305793e-08]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [1.1772661561235415e-05, 0.39935602202777587, 0.88471442533348]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.0703713744529558e-05, 0.39722398635297496, 0.8974541650769731]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.6463492449047907e-05, 0.38814135753461565, 0.829830053744367]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0003631080421416811, 0.34531074549578555, 0.06744240076116119]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [1.4063701640097907e-05, 0.3918240134316281, 0.8583660435868774]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [2.3539822987754957e-05, 0.38179651994460484, 0.7507397763605347]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.0501051083343113e-05, 0.3892571439587205, 0.9010602897589732]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9753684103140193, 0.21518661964980587, 1.0008647942287686e-08]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9758517685989823, 0.2143837765480473, 9.734206338999416e-09]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.345450810962778e-05, 0.38105828911155726, 0.8667933194957916]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0011163279873296762, 0.32317538145463887, 0.0155764580150902]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0004376959977317426, 0.33859663942698764, 0.05366209085452085]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [1.0590064841048075e-05, 0.39051781394632556, 0.9003300436616831]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9737179390987828, 0.21650967734884596, 1.0992467129672047e-08]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2272193124841835e-05, 0.38443462375122883, 0.880951075835635]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0006341534243825273, 0.3318803453930698, 0.0331667523263713]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9736435656938585, 0.21689302463448698, 1.1056089172391069e-08]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9755133830623728, 0.21554535318183565, 9.960722138928322e-09]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.0905110116930763e-05, 0.38579639709732094, 0.8969942599247311]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [1.7792682801381738e-05, 0.37627488363146394, 0.8173697716847995]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.0386614051834232e-05, 0.38013180873407876, 0.9034439525442507]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00014758005335964637, 0.3439077417447472, 0.2011389019146153]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9755082554103429, 0.21289258551936752, 9.978070458826388e-09]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.001714046564228717, 0.3179005474010813, 0.008776199207072287]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9737851659717001, 0.21544110950992698, 1.0985683732956373e-08]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [3.577199502930644e-05, 0.3689057083285885, 0.6336434046989373]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.3459070192982568e-05, 0.3864848180293022, 0.8651356360029799]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.975284954726675, 0.21518466642362585, 9.972970965385827e-09]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [8.380095644105524e-05, 0.35898458066342637, 0.34745375393904604]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.0893897163050743e-05, 0.38239261981877815, 0.8975343547755812]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [1.2737260311837083e-05, 0.3772199838728452, 0.876368666024336]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.1905922017410351e-05, 0.3750022569588919, 0.8861954995045421]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9740824868915043, 0.20918500032970896, 1.0867066737182355e-08]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [5.517981660760032e-05, 0.352075189345706, 0.49108996340012884]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.002530247821397368, 0.31062349725711125, 0.005129049698393556]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.002166719067866395, 0.3162626626927072, 0.0063312211681776755]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0006964497695225644, 0.3338634790514072, 0.02909377968692632]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9748610900803599, 0.21826546306916625, 1.0282991839430819e-08]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [9.478642659991533e-06, 0.39212299607143203, 0.912712355741687]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0007606785368490597, 0.3332767589683736, 0.025906387018653717]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.003212026239800784, 0.31981446671579444, 0.003710275230605723]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.00031561391459653627, 0.3525462565664822, 0.08105717670845644]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [8.658851376866004e-05, 0.37371627328284385, 0.339315997117879]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [1.540020772083802e-05, 0.40155882065594417, 0.8422900317833579]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.4532046401142154e-05, 0.39894232820509384, 0.8528989984084486]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9753816494524078, 0.22266045767128026, 9.949025862009201e-09]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9756210739901197, 0.22189658552747793, 9.813742804524248e-09]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.489574144775864e-05, 0.382138573421266, 0.6380637574915964]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [8.267177860271882e-05, 0.3679242998265273, 0.35515646882284746]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0006108368761638731, 0.34685717581284825, 0.03446076904614549]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.001649954246611409, 0.3384756966797221, 0.009125650018375469]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9748775569916446, 0.22768341726084618, 1.0235279658934313e-08]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.654852162999735e-05, 0.402362452199759, 0.8297413577672115]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9754756184797908, 0.22503982360105637, 9.919257096279127e-09]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.00020961077786668965, 0.3643389295535088, 0.13333762433367394]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [2.2246140674185205e-05, 0.3988850808138643, 0.7653402754447851]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9755754131523163, 0.22521066272283274, 9.88694616572797e-09]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.012191736969837122, 0.3134625020604824, 0.0005984482265971109]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.003330527967426736, 0.3333302147417791, 0.0035387470017439783]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9759610342397028, 0.2295579774827492, 9.6704006794126e-09]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [9.994044449257673e-06, 0.41367335596966137, 0.9070917659165081]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9748786639002442, 0.2275399338056443, 1.0305525627814223e-08]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [1.2200505978847001e-05, 0.4063464479511526, 0.8816419740303494]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.3999936422841225e-05, 0.40098682982130096, 0.8608630628648469]

Epoch 5000 root-mean-square error: 2.919905910154676

Epoch 5100 root-mean-square error: 2.9232593434513077

Epoch 5200 root-mean-square error: 2.8567749341879436

Epoch 5300 root-mean-square error: 2.9282569410917585

Epoch 5400 root-mean-square error: 2.914270797317124

Epoch 5500 root-mean-square error: 2.8764532813745687

Epoch 5600 root-mean-square error: 2.9264815183831296

Epoch 5700 root-mean-square error: 2.924780931326307

Epoch 5800 root-mean-square error: 2.9391107671898795

Epoch 5900 root-mean-square error: 2.9302511190270413

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.980209306414389, 0.21029903389254845, 1.9311562922184794e-09]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9801455208369462, 0.2097474946928371, 1.9392130432108697e-09]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.01682810343143533, 0.3055251051811509, 0.00018888053109626006]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0008373142098786441, 0.35182283371433565, 0.014420669960904475]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.0006185737082463194, 0.3605399247526467, 0.02214617710542715]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9795952029185909, 0.2168994793259757, 2.0185752425494453e-09]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.3734799618179724e-05, 0.422789406624621, 0.8459264707407713]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0007071631862325457, 0.35877943976697807, 0.018377025385829842]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.2059718755285081e-05, 0.4259995215459096, 0.8691465277207406]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9789324789954498, 0.21564635534603324, 2.129054079363124e-09]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0005550923856215476, 0.3625716384278193, 0.02591396761648549]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [1.819670648672238e-05, 0.41972457463581436, 0.7862415568685605]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [6.245341969899873e-05, 0.3969170480447193, 0.3844775472787365]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9803516254029034, 0.21743686120989333, 1.9061859929565587e-09]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0002874142841845539, 0.37725970783980817, 0.06389337739324101]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2390069949067358e-05, 0.4312199025061472, 0.8639428251516815]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.0392540354086669e-05, 0.4301844170828377, 0.8913189353754867]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9793852800078179, 0.21625313782479125, 2.052196780101511e-09]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.007405789351555927, 0.32620048559639064, 0.0006254987071367144]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9753039648350692, 0.22023332343499583, 2.6856282868346734e-09]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [2.255325502472617e-05, 0.41688018509680896, 0.7288541036685363]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [1.078163655794062e-05, 0.42493646659526624, 0.8870435224194081]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [1.1917276817442847e-05, 0.4195970521978007, 0.8719276325132785]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2255054879669697e-05, 0.4154808874641196, 0.8676018531081124]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0008574411840501059, 0.34829518848353824, 0.014129161297427112]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9795439918262848, 0.21304610950319403, 2.054310320103487e-09]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9786371699734562, 0.21288941569186792, 2.1835861513746877e-09]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [1.0766628322858908e-05, 0.4169649808101923, 0.8878540824649601]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [1.1779185005245524e-05, 0.41191827778083884, 0.8744715994558271]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [9.923648429521545e-06, 0.41096390690748735, 0.8993761482249585]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.979390246655453, 0.20691450748138598, 2.078721762740236e-09]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [2.476679105663032e-05, 0.39273148471793967, 0.7052954472505198]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.1111473734404067e-05, 0.409941966686075, 0.8815574894152819]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.980757246730911, 0.2063682734273155, 1.851680043990521e-09]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9799565625404685, 0.20625566547058835, 1.967909157124592e-09]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.4272221628455509e-05, 0.40095377105921565, 0.838658304056953]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0066678227639872115, 0.30956629969351834, 0.0007302561101658875]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [1.4187356866838197e-05, 0.40217502736310967, 0.8402453608378038]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9803599243399227, 0.2047437944675957, 1.9176970449844148e-09]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9798287130440222, 0.20448740467102308, 1.9958037840346594e-09]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9802979462385611, 0.20369644535148862, 1.927035789037204e-09]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [1.104861406194429e-05, 0.40013580237400176, 0.8832884423228545]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [1.1232212088301542e-05, 0.3964678828087863, 0.8810022470365817]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9787474874813511, 0.200852602699499, 2.1590650134744075e-09]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [3.441859299529892e-05, 0.3758618746667587, 0.5960397148130211]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.2717144630254549e-05, 0.3874614772489751, 0.8628173622757913]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.1177070965159825e-05, 0.38612601244598743, 0.8836525560966954]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.2271004042911795e-05, 0.3815560544292297, 0.8692761672771391]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9780208157020888, 0.19491468217826358, 2.3010714554326574e-09]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.000987100346557363, 0.3170718662640051, 0.011734137848354293]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [1.1672198975018124e-05, 0.38309523775459986, 0.8775016588068932]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9780888038787698, 0.19531305161622795, 2.2950398098243376e-09]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00027013990440964214, 0.33516675167882665, 0.07165911835035808]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00010916360114748169, 0.3520894962227832, 0.22187726379822123]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9792742721149508, 0.19898141874611172, 2.1071011544176836e-09]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9803415251799276, 0.1979285246427095, 1.9466257987303868e-09]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [6.316748166770645e-05, 0.3630047049622046, 0.38448022904345064]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9797147986296871, 0.20009878002648201, 2.0207505908898205e-09]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.3090068555969744e-05, 0.39003228435699444, 0.8565295507032726]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9804544730771032, 0.19773465390033507, 1.917017406710987e-09]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.011809683638451581, 0.29181747872688923, 0.0003216120219576235]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.002319290246203288, 0.3169300972165756, 0.0034000479112720424]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0016873468499188705, 0.3251658788098948, 0.005374118728826999]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0001808910937569644, 0.36086463659428775, 0.1193888531441239]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [1.8765139980145446e-05, 0.3992102964794703, 0.7804440797951663]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [1.0357833850301191e-05, 0.41338916602311687, 0.8914464196969716]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.003930809758761254, 0.3226791094072717, 0.001560382295563407]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00030204389225603305, 0.36346207130745106, 0.059732896232436015]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9804147478216629, 0.2123809728225908, 1.8950459397143527e-09]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9797888554451282, 0.21214569507811631, 1.987385405116887e-09]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0001337171997927333, 0.3784453264879712, 0.1705906593499383]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.8827773820634775e-05, 0.41331614365076874, 0.775883402294748]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.994833428405137e-05, 0.39817862472748605, 0.540612704398435]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [9.078820493638777e-06, 0.41777153050293575, 0.9101719140012465]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9804316834139719, 0.2086829360414799, 1.9193164667938274e-09]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9786452017403637, 0.20906749732640706, 2.1860390170586944e-09]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [2.213629374820067e-05, 0.39875319939446474, 0.7372189436236048]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0033720387064712877, 0.3292850961119659, 0.0019451343082641777]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9804130844131655, 0.212408530986405, 1.8950646235750975e-09]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.001663204102595786, 0.34258066613300286, 0.005384377403441723]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0002499061435098995, 0.37488704900968267, 0.07699198742550271]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9801302913371048, 0.2169381325572082, 1.9344563058896108e-09]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [4.644934706576944e-05, 0.4045387327652519, 0.48541785455847564]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [1.539103003596326e-05, 0.42693827833025116, 0.8200825074607391]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [2.3725912964178314e-05, 0.4163453386367509, 0.7103160783421348]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [4.226604245914737e-05, 0.4037863032662755, 0.5182424869607966]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00046349177854140085, 0.36408786449014124, 0.03343566466746312]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9800742160692723, 0.21613770538069868, 1.959043591721629e-09]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [6.466416073440023e-05, 0.39773517133772995, 0.37205712701985183]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [1.0556768348372926e-05, 0.43122354245437683, 0.8887340470767949]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.2321337953092707e-05, 0.424913498568071, 0.86490343118573]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00031871574716439016, 0.3710163898342368, 0.05563246670983339]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0006976470211412647, 0.3636305394548132, 0.018673112778275408]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9797577731708645, 0.21960298990898391, 1.99412676899705e-09]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9790792379878226, 0.2193368232570662, 2.0937708956694156e-09]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9800674305415238, 0.21813751458383443, 1.948687777415381e-09]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.4762995355757094e-05, 0.42533898352258687, 0.8317704867982063]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.017867423335067478, 0.31475731572258625, 0.00017330747384655868]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [1.197043283365312e-05, 0.42945129512178004, 0.8703178424697863]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9794687011027817, 0.21681052775555437, 2.0468666376140227e-09]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.1879492415139667e-05, 0.4248570932935838, 0.8718067996982816]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.048258413201362e-05, 0.42306969267488154, 0.8908566700159557]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9806512321321046, 0.21201535843406374, 1.876398259914074e-09]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9807606038470663, 0.21136010701262575, 1.8604990909149815e-09]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.0332411989227414e-05, 0.4178260183793805, 0.8930238651355198]

Epoch 6000 root-mean-square error: 2.911909177760193

Epoch 6100 root-mean-square error: 2.9044084106307153

Epoch 6200 root-mean-square error: 2.9284596863575496

Epoch 6300 root-mean-square error: 2.8911674150679842

Epoch 6400 root-mean-square error: 2.9211856414380093

Epoch 6500 root-mean-square error: 2.9057632048083706

Epoch 6600 root-mean-square error: 2.9275002730293775

Epoch 6700 root-mean-square error: 2.893612787843854

Epoch 6800 root-mean-square error: 2.853367853211478

Epoch 6900 root-mean-square error: 2.878706094225238

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [9.307280306155414e-06, 0.4093795951860668, 0.8961310512343353]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00018845426760004722, 0.36010858580891325, 0.08215223242195249]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9822736284460171, 0.2025391610498804, 4.4121088172461374e-10]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0004354172221473002, 0.3513983560497941, 0.024449488529568814]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9820886122657303, 0.20451590376591217, 4.4823793458498146e-10]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9820078126995125, 0.2040293328628736, 4.516125155100227e-10]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0005013366840569493, 0.3520923379501015, 0.019818811594363574]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [2.1151267809265678e-05, 0.4051875848195153, 0.7126882615922339]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0007809709785241151, 0.346810390365165, 0.010281265030011675]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9821042109689601, 0.2066607125746287, 4.503026950504156e-10]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.981190595931034, 0.2066725504388289, 4.873891309495486e-10]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9819740386355573, 0.20563533084893273, 4.55583676796887e-10]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [3.717648319484667e-05, 0.3952472254066923, 0.515049199149898]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.006386353885233993, 0.3220322324618134, 0.00041598630738315964]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [5.759085215399894e-05, 0.3976492600294115, 0.34914999675896397]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [1.9716090493164268e-05, 0.41970880674047273, 0.7299380941077677]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9821369117749598, 0.21050743245779022, 4.4261600019079855e-10]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [1.682006865407788e-05, 0.41782363215401275, 0.7761392040685735]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0006124860251340477, 0.35810557786005304, 0.014590753804289466]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [1.4589829184480482e-05, 0.421355787797369, 0.812299336458706]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.4678916307923488e-05, 0.4175982742896709, 0.8115247599660488]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.010397098600451968, 0.31451537738256496, 0.00019904073402943277]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [1.0126025882692622e-05, 0.42449016313379845, 0.8837288394165197]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.3137235242368743e-05, 0.41657611328312605, 0.8367866003641916]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.0598236749437396e-05, 0.41636766536456615, 0.876969819741349]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00020202451825995248, 0.3672080175145119, 0.07496717789903581]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [1.6756813722632895e-05, 0.41043132627718376, 0.7806211079654767]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9820819388657483, 0.20950876372220492, 4.448845559345305e-10]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.02658112458346133, 0.30336798203476395, 4.613525660526058e-05]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.003370420771485194, 0.33692276555234046, 0.0010997167432620985]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0025971757822157693, 0.3447765216008352, 0.001634827971397975]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [8.516384126933311e-05, 0.40213132066898666, 0.22799134584804034]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [2.1834675296978907e-05, 0.42901371918500736, 0.6989019344647642]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [1.579489111827231e-05, 0.43059825851798883, 0.7929920499062947]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [1.9404842770285097e-05, 0.4234912518194245, 0.7379466298454466]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9825780760018649, 0.21315795924180497, 4.301902294448733e-10]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [9.31369524209483e-06, 0.4308186529253523, 0.8964041612691704]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0007035759077472199, 0.35930138311613885, 0.01198634438404373]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.1054643073631756e-05, 0.42908360363305015, 0.8697747174650464]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00023233107564807963, 0.37724838219491386, 0.06145930670210546]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [8.49806221036478e-06, 0.4345027425354651, 0.9088972070416559]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9806800555978775, 0.21332298206718073, 5.063629852727328e-10]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [0.00019361214834584233, 0.3802350387417251, 0.07956392713139715]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [1.942988502442326e-05, 0.42133009904459484, 0.7396607639551256]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9819251794827957, 0.21682793166678116, 4.503902186356052e-10]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9823980047362137, 0.2158939626537104, 4.320978919530641e-10]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9814454419652242, 0.2158858695270364, 4.690560372289693e-10]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.982419763365862, 0.21464297178657915, 4.311898334646019e-10]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [1.0231224681897539e-05, 0.43318221088466935, 0.8805811953201553]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.981441384473902, 0.2128663301136838, 4.697389309219855e-10]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [1.2757106078172825e-05, 0.4248615820401603, 0.8408704923959536]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9814993792429483, 0.21050389370709024, 4.686455164643711e-10]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.00015134883105819182, 0.38117996957813755, 0.1101326144462189]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9818196145218748, 0.21216653779249686, 4.55880371970071e-10]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.6364244913970846e-05, 0.4203574644152095, 0.7838433779396653]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.000126776737271968, 0.38433146458654865, 0.1398460759744718]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9826780790662176, 0.21173494146100974, 4.232116886869397e-10]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9814053084452447, 0.2119583422220383, 4.728757601254027e-10]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.1429938225132405e-05, 0.42575484658860235, 0.8625239628058088]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9821591775939001, 0.20916676729908157, 4.440165021227038e-10]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9814570589028542, 0.2090465818300093, 4.723311302988486e-10]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9818759802578976, 0.20819473652401235, 4.5477480983875173e-10]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [1.1624667225142533e-05, 0.4192021795454943, 0.8597607900004071]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.2891340908219273e-05, 0.41388832824865934, 0.8400616653436528]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [2.4999278840634687e-05, 0.399879323090932, 0.6582920424923545]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9764892758430834, 0.20560454636038353, 6.912980683994585e-10]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.1929015444220137e-05, 0.4073543696503787, 0.8570689562838399]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00012239443772474199, 0.36812929893254065, 0.14877561718013807]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [9.696173925657557e-06, 0.4120968478710181, 0.891468862751305]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.792619370502646e-06, 0.4100348561549329, 0.9051652142554197]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [2.6519621638512464e-05, 0.38938705934593276, 0.6410031292385544]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.010264344646556399, 0.30641822087603326, 0.0002017499707306642]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [1.5917865999820645e-05, 0.4067485941037544, 0.7915077469442181]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [1.2001858570303226e-05, 0.4077205792458066, 0.8542135147118232]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.3527828892494918e-05, 0.4023579073102581, 0.8304604845175734]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9815332634469351, 0.19999283653321218, 4.717991566300802e-10]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9825465694245837, 0.19887646111633697, 4.3178544705846774e-10]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0040273449964547915, 0.3137083712617474, 0.0008515919627435273]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9827056026492648, 0.20063374616827287, 4.2562945865928765e-10]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [9.676646086819443e-06, 0.4064426243000784, 0.8910537430966787]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9823297149368407, 0.19878191068848727, 4.4067859467804085e-10]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9816877536006781, 0.19865114693753325, 4.657488482122183e-10]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.982106780384327, 0.19790470842049732, 4.492624087818877e-10]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [2.1805502084842386e-05, 0.38817321249309633, 0.7045101985565508]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9823794458524853, 0.19579145818899787, 4.414663514295305e-10]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.0650540660013704e-05, 0.395209773977935, 0.8772481940024262]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9814235667443075, 0.19437272615294973, 4.801880065775788e-10]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9817336894025305, 0.19375648188997263, 4.691261286300691e-10]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [1.7452108570793454e-05, 0.38295631734431596, 0.7717497344795633]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0011888473453029026, 0.31906149696703884, 0.005526437195311517]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.0002739891419642916, 0.3438514111158497, 0.049226323174408754]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9813903925609482, 0.1966702098484431, 4.832294964937553e-10]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [8.145700434103825e-06, 0.4002234148113231, 0.9154236632297073]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0004244212623350172, 0.33820334071696007, 0.025942587706034515]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [9.659745146118027e-06, 0.39902620831618213, 0.8931747649552959]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [5.671462038527246e-05, 0.36892010592154684, 0.3623935351620255]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [9.528366150039271e-06, 0.4008719295791425, 0.8941548017549571]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0006695182498099775, 0.3344060605047387, 0.013034569925572215]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0027079257568068515, 0.31871312477256253, 0.001572631043584209]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [9.55021233754814e-06, 0.40689276609175284, 0.8939618404116711]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.001089723155244022, 0.33265712673704484, 0.006264905819172412]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.980495189298515, 0.2023065520735103, 5.169857315250692e-10]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9821992281056692, 0.20080884884478356, 4.486326499764615e-10]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [9.362061132467166e-06, 0.40689898506539995, 0.8969426590995389]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.01054799276546628, 0.3006581532370908, 0.0001976181288440514]

Epoch 7000 root-mean-square error: 2.8884891227243927

Epoch 7100 root-mean-square error: 2.8836603516426536

Epoch 7200 root-mean-square error: 2.8915221758974465

Epoch 7300 root-mean-square error: 2.9227078617098194

Epoch 7400 root-mean-square error: 2.9002670587385326

Epoch 7500 root-mean-square error: 2.882854614319121

Epoch 7600 root-mean-square error: 2.874460749726691

Epoch 7700 root-mean-square error: 2.889867357021718

Epoch 7800 root-mean-square error: 2.9102535242958956

Epoch 7900 root-mean-square error: 2.8768261921157037

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.1655216430824656e-05, 0.4148976450616841, 0.8436733712591902]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [9.320054412975286e-06, 0.41496095304296293, 0.8854255520748145]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9832356958704, 0.1959754927201133, 1.1574673350604808e-10]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0006173024256282849, 0.34440338491705774, 0.009648764984337225]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.4242369872369512e-05, 0.40850509826401593, 0.7976021743624189]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [2.8027147821889713e-05, 0.39409005833882005, 0.5739915051586275]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [6.788661613974538e-05, 0.38473453874240016, 0.24411169118329382]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9841367762614003, 0.19650754980970486, 1.0609089956636403e-10]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.149580177459356e-06, 0.41531445054156396, 0.9059068078759658]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [8.375383389047814e-05, 0.37434072137729696, 0.19085967261761805]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9840326162158214, 0.1968114819215716, 1.0700045743562957e-10]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [8.146056463854711e-06, 0.4159667266444724, 0.9057137720150423]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.0463785026626945e-05, 0.40822419208585703, 0.8658682481997493]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.0046672659550487e-05, 0.4053564831157407, 0.8734672420127296]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00016240600723188548, 0.35820744188003045, 0.07601780782916667]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [6.516383616328928e-05, 0.376983755435734, 0.2603591307941379]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.02200325911864864, 0.29396002239368063, 3.17788970244457e-05]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [1.5059967401002727e-05, 0.4096675872217319, 0.7827073826508485]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.00820918763063992, 0.3156156201262359, 0.00015352786838539694]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9833004333558413, 0.20376680433596148, 1.1338103064597624e-10]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9835976309999509, 0.20303250314295168, 1.1017332543257632e-10]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0019092647457431093, 0.33998084168309, 0.0015787694590432208]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [2.110911624749402e-05, 0.41676123649999325, 0.6736189204851623]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9841618524077244, 0.2028557941748608, 1.0476594460070225e-10]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9839526507648569, 0.2024748194954691, 1.0703394853048629e-10]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.036398446782346e-05, 0.4234571646623681, 0.8661799188244501]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [8.87242873921626e-06, 0.4222984945572869, 0.8925728934063868]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [8.791286776323623e-06, 0.41870649565567924, 0.8941132812308058]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.0004522960293848018, 0.35232494213623156, 0.015674519662768144]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [9.5534461003454e-05, 0.3811602560315516, 0.159271203053317]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0007013939661503961, 0.3541863590183383, 0.007832180671430655]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9825937422532505, 0.20503607404040164, 1.2241320534831376e-10]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [9.700158460818181e-05, 0.3893535219498369, 0.15565414851305276]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.983543903851876, 0.20624958071211105, 1.1179835081068678e-10]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9837051300258516, 0.20554957238474306, 1.0964345035546024e-10]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0013998501021381666, 0.3495693667478038, 0.002610361292662024]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0006501291931056107, 0.36593137550154237, 0.008806815871057881]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [1.0940512065897675e-05, 0.43847546583205105, 0.8558076754216926]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [9.02662784359125e-06, 0.43781979450189296, 0.8898974445290131]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.0929930926361861e-05, 0.43064423983571987, 0.8565416005605848]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.639639770744748e-06, 0.43076206748260915, 0.8969416901068575]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9821260899396144, 0.20389434837372208, 1.280984733773022e-10]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9836076473118484, 0.20239217092152034, 1.1139879882811798e-10]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9830557120692796, 0.20225107806596335, 1.177158174353621e-10]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [9.517077353043578e-06, 0.4228781331585099, 0.8819716446524722]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9775180204538925, 0.20322666641781265, 1.8679831925198575e-10]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0007000852436519725, 0.349503100843489, 0.00790813292727779]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [8.543532314335309e-06, 0.4250040095081393, 0.8988910375549325]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [1.5134120827155121e-05, 0.41179160074981436, 0.7817465493243282]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.982889158042731, 0.19873779447990983, 1.2014434016232792e-10]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [1.051137118959359e-05, 0.4134626211329813, 0.865504234065574]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.00035305808040795347, 0.3541634121774529, 0.023368500093041877]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9817094126308313, 0.19972706817109107, 1.3412299396815172e-10]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [8.486249447906912e-06, 0.4175146402778754, 0.9006880897342494]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [8.209951007138188e-06, 0.4143705388135283, 0.9054192466090294]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [3.070525225036168e-05, 0.38953368798865007, 0.5397982793038126]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.00381716658702583, 0.31998363861660295, 0.0005288561512146386]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9828929272120955, 0.19995224366161615, 1.1910306039627932e-10]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9840983164234937, 0.19861624962061228, 1.0557718744808352e-10]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0007580401922513603, 0.3468082928380886, 0.006925996986298651]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9820649843659806, 0.20181428707853272, 1.283015124767754e-10]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9826777047171773, 0.20091273768822882, 1.2136799002441137e-10]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9833501071189308, 0.19995886508002053, 1.138531241159093e-10]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [1.0438091254499504e-05, 0.41753305647667777, 0.8651323679380482]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [9.332794423140916e-06, 0.41571754012070455, 0.8848411028780783]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.983211447022651, 0.19631942633538663, 1.1575513009566791e-10]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.1700669613815646e-05, 0.407638429000359, 0.8430184344096515]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [7.872517272949573e-06, 0.41055341867134054, 0.9101041416367991]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [2.6060052867911283e-05, 0.38782705643832643, 0.6010660227388492]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9839561002424503, 0.1945134504657819, 1.0626959672090034e-10]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [1.1772205116886458e-05, 0.4048251154994903, 0.8394975831942886]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [1.569927446628786e-05, 0.3967499839523256, 0.7684094859269306]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.001527566651548422, 0.32350979391658824, 0.002271638387200848]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [2.8534368449086513e-05, 0.38857840880733857, 0.5631715851601056]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9828041494426761, 0.1926261829970982, 1.2087265909888162e-10]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [9.688951071348193e-06, 0.4019233529948272, 0.8796807794998518]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [4.398907232509305e-05, 0.3746889102344379, 0.39698111532283187]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9837484816970622, 0.19233920298403212, 1.0937141168447708e-10]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.4364706799716116e-05, 0.3965171510211264, 0.7942416738524555]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [0.00015808959960851248, 0.35589449630677744, 0.07845701978332364]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [3.6667893723766325e-05, 0.3832039287052355, 0.46570990172220855]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9841384729478105, 0.19472725469409768, 1.0433646944713046e-10]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.02332129959825346, 0.2902655441563291, 2.8442879930242046e-05]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.09875690586622096, 0.2727984953586586, 2.514752889745964e-06]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9842717582444626, 0.19873060144543142, 1.0272777939517886e-10]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.010713718310092374, 0.30755403160057515, 9.99709427262894e-05]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.4619422421864532e-05, 0.4143439455762495, 0.7872614071816252]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.0047733290776386305, 0.32032611774846587, 0.0003671700384603004]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.027232124122088e-05, 0.4212264103348356, 0.8670967813339209]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.2153557325856738e-05, 0.41476082023988226, 0.8334225056366927]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9839848952621161, 0.19828762588618148, 1.0656507455754594e-10]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9845531580221903, 0.19738943513204393, 1.0045757398370674e-10]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [1.110246147700047e-05, 0.4111202026831472, 0.8529026005593026]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0027654318994631234, 0.32222470370578093, 0.0008844761117713125]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00016108888046633744, 0.3694649652749355, 0.07594166445005326]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [1.1822784564086754e-05, 0.41619037493503824, 0.8401632723545202]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [5.968023740971233e-05, 0.3864457030565775, 0.2859267357827636]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9841565537710164, 0.20099632316952157, 1.0444675333014796e-10]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9842935519816812, 0.20038587079892936, 1.0302311224162779e-10]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.0007666498209317673, 0.34943861032719614, 0.006766248101422272]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9842197796486599, 0.20229565580271705, 1.0375335324096136e-10]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9845390999399751, 0.20154226309509526, 1.003841470597141e-10]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9844945471637858, 0.20105482619040735, 1.0087340052617568e-10]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9842465855119853, 0.20071590962763305, 1.035319833310811e-10]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [9.651351768987568e-06, 0.4210896115035867, 0.878428085691836]

Epoch 8000 root-mean-square error: 2.9141145349178856

Epoch 8100 root-mean-square error: 2.87959251920194

Epoch 8200 root-mean-square error: 2.896872696222803

Epoch 8300 root-mean-square error: 2.897251539226684

Epoch 8400 root-mean-square error: 2.9142645525699824

Epoch 8500 root-mean-square error: 2.880068339359305

Epoch 8600 root-mean-square error: 2.888306995840898

Epoch 8700 root-mean-square error: 2.8972740000169823

Epoch 8800 root-mean-square error: 2.895907405009163

Epoch 8900 root-mean-square error: 2.8895150721019194

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00021421547135045007, 0.3611524801149556, 0.041999693318012704]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [8.741789210261817e-06, 0.4185696240000285, 0.8997731150372453]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9858283462664857, 0.19106194495750722, 2.975078676818948e-11]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.0787657169413854e-05, 0.41063029531245837, 0.8636854696724456]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [7.952437843412867e-06, 0.41215436833316904, 0.9134022903657448]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.0004069289037806713, 0.3452712218996108, 0.014909748832413902]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [9.873431201495253e-06, 0.40985310899812744, 0.8804735114170487]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.985445646812456, 0.18857807081175124, 3.123622359030386e-11]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [9.88072935149484e-06, 0.4055867992335506, 0.8805443368595414]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9862317019273048, 0.18605549047979328, 2.8463579590946605e-11]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.1188017332760919e-05, 0.39937805517015856, 0.8572850244521595]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [1.2126122646280288e-05, 0.39467439097515167, 0.8404190981337517]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.686711189279886e-06, 0.3967566906447396, 0.9019915094904233]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9803277231994563, 0.18504939857691163, 5.247107477187675e-11]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.1071303188323534e-05, 0.3888004122483704, 0.8602408026674055]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9847756173114569, 0.18059620069838506, 3.397928646425607e-11]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0003231181276330726, 0.33285926566982016, 0.022001405412670358]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9851083037230948, 0.18217130620259386, 3.270267562629235e-11]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [9.411314354210367e-06, 0.39172419274975656, 0.8899390100538473]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [8.445732756275637e-06, 0.39014869630428256, 0.9065172533180625]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.00013268847918047845, 0.3441386540123047, 0.09028834371542295]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9838449977970173, 0.18202727412337924, 3.7578681663883685e-11]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [9.704256450925588e-06, 0.38894003907574354, 0.8850252663275768]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0001384678371352532, 0.3444118779007732, 0.08464816938874335]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [8.840262706174634e-06, 0.39211462529197755, 0.8999669717275349]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9846876384508931, 0.18053785691126858, 3.445037111108744e-11]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [8.139669182570707e-06, 0.3894756614204024, 0.9117862773401756]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.0418089006379638e-05, 0.3822810098993382, 0.8728072948985047]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [8.521118092489032e-06, 0.3822805006227018, 0.9057308854485138]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9852551014912168, 0.17574487421612947, 3.236956672507187e-11]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [1.170298078012263e-05, 0.3735651395601687, 0.8501509608539465]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [3.171110966648754e-05, 0.35533971504498046, 0.5198986948287081]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.2066894136718548e-05, 0.375066192418556, 0.8414330486566592]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [4.343847810880495e-05, 0.3524590021457549, 0.38714695060743737]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [0.008887209137230347, 0.2804107628200018, 8.787956694109211e-05]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.98642040531876, 0.17842659592521212, 2.7641679975902844e-11]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.986326759768995, 0.17811556227382735, 2.7975201502638563e-11]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.06445957668449455, 0.25595277422128837, 2.9555976662837733e-06]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [7.116964985735037e-05, 0.3564406212210976, 0.2151198543194863]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [9.101705377032308e-06, 0.38587751020283656, 0.8953114529914348]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [3.0178582602421257e-05, 0.36400352236773603, 0.5381954435319624]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [9.24743842458303e-05, 0.35158052788021815, 0.15071227419983438]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00021600406929581976, 0.34317818738321126, 0.041355850053404995]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9852240479596215, 0.18462565976262862, 3.181891380120216e-11]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9855288277025814, 0.18399414038433148, 3.0723908145893915e-11]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [2.114279868883207e-05, 0.38274373149710317, 0.673295058909287]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.400605172902324e-06, 0.3944367989206982, 0.906293749190514]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [8.910592353924813e-06, 0.3901349798480879, 0.8977368871670524]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.985582710827809, 0.1794431547683731, 3.090450167338134e-11]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9863908795388975, 0.17844616909630454, 2.7943032798577232e-11]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0006640534727549152, 0.3199287501702131, 0.006689436633005828]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [1.0168845333104947e-05, 0.3882618689000053, 0.8758843682296245]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0006653097843398302, 0.32124405082861607, 0.0066806804997511255]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9859453973275105, 0.1813944175130972, 2.955303098862715e-11]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9859909262591023, 0.18096006901803222, 2.938622242618827e-11]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [9.607003448769043e-05, 0.3532872638211511, 0.14420132257046245]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9852841509744042, 0.1832793521241745, 3.191477345142361e-11]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [8.040804746256265e-05, 0.36001642651366716, 0.18437681921047033]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [2.87106307032902e-05, 0.38104324788580496, 0.5554433097805432]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9863740692455314, 0.1864823002195624, 2.753708604743848e-11]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.0982706735234717e-05, 0.4009685438796214, 0.8583552962335005]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.1897060544858779e-05, 0.3962480182471012, 0.8417377951662705]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9857371200105854, 0.18365396140385593, 2.9880059719041255e-11]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9857361306872116, 0.18324734926921343, 2.989892699623838e-11]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0013741614902167316, 0.3180119941082582, 0.0019687921157324115]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [2.5806656388156678e-05, 0.38373903879077287, 0.5953785505865107]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [3.106285937968637e-05, 0.3856558903911693, 0.5143497167659561]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0003901855866775192, 0.34265537900541665, 0.01574213334180255]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [1.3654481557307968e-05, 0.4007852614561522, 0.8087465629164906]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.009076062870176406, 0.2970579924834989, 8.432680532048303e-05]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9860276342611457, 0.18868183039555445, 2.8874471466240196e-11]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9861818588338067, 0.18811543495414262, 2.8320031154775555e-11]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.001956152366554048, 0.32206112370709616, 0.0010949832850945414]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [1.103761008174668e-05, 0.4087176937513814, 0.8581751322453246]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0018957406282233752, 0.32361302300962796, 0.0011563581028899669]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9856270820099888, 0.19114152099718418, 3.035261330598927e-11]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0005029750790305247, 0.3473743474111882, 0.010439068658230723]

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9864597657454433, 0.19231925281447643, 2.7415097026700897e-11]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.985342939503421, 0.1927314692884087, 3.137214126940909e-11]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [4.057687787185424e-05, 0.39095579375670775, 0.4102405975158406]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.020240362781172527, 0.2979052997951383, 2.157744571828262e-05]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9861627749145518, 0.19619466819677103, 2.8162585767994827e-11]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [3.1070353113819955e-05, 0.4038727415359955, 0.5167535586141009]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.985265723836815, 0.19489389092927079, 3.1753318170804237e-11]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9853473483376939, 0.19434229835120784, 3.142235101478672e-11]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [9.036917717142579e-06, 0.41988421843008444, 0.8946139120224658]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.986467130664388, 0.19139209139519514, 2.7478032277631e-11]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.003060942085834133, 0.32180259984699355, 0.0005226659926954133]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [8.797703948333612e-06, 0.4206313952896065, 0.8988891918950368]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.985852438173311, 0.19212580933640033, 2.9654996744811374e-11]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [9.554518706275699e-05, 0.3766579896501289, 0.1440597080269409]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00030671110206934374, 0.3623454357745098, 0.02354261285478589]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [9.521402696359847e-06, 0.42485615764498313, 0.8861694346857966]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0001135064672750926, 0.3796339515823805, 0.11209969470153133]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9845763330071498, 0.19791100194664, 3.427791176329112e-11]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [1.6004269404742155e-05, 0.41647893544688486, 0.7664586650775163]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [1.300730364738307e-05, 0.4250746579951267, 0.8195468729882356]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.0007389962734029913, 0.3541405776053987, 0.005467015347037019]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9849828374398948, 0.2000708377155938, 3.239300507296756e-11]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9861330473814177, 0.19864722594194426, 2.831108620243993e-11]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0005773995233003446, 0.36094308529367747, 0.008222029262117589]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [9.353875793856543e-06, 0.43527010119442633, 0.8875246400253143]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9857042985188579, 0.19913125776874235, 2.9827066183727633e-11]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [8.823708962778185e-06, 0.43158833775829664, 0.8970129854381679]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [1.0146854271884481e-05, 0.42531708095971105, 0.873634017726758]

Epoch 9000 root-mean-square error: 2.9067049726784413

Epoch 9100 root-mean-square error: 2.8745513180954463

Epoch 9200 root-mean-square error: 2.8696879704358045

Epoch 9300 root-mean-square error: 2.860890409006834

Epoch 9400 root-mean-square error: 2.883055803785715

Epoch 9500 root-mean-square error: 2.865555405192009

Epoch 9600 root-mean-square error: 2.9117404122020956

Epoch 9700 root-mean-square error: 2.8661964734680843

Epoch 9800 root-mean-square error: 2.8634209223973843

Epoch 9900 root-mean-square error: 2.899036877729297

Training Information
-------------------------
input values: [5.5 4.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9875072501986063, 0.1875620658912857, 7.615487618275909e-12]

Training Information
-------------------------
input values: [5.1 3.8 1.9 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9865484385686659, 0.1879495211464508, 8.67000953418651e-12]

Training Information
-------------------------
input values: [5.1 3.5 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9871473828190476, 0.18700998988759387, 8.007930449444347e-12]

Training Information
-------------------------
input values: [6.7 3.1 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [8.679989696813517e-06, 0.4203305738594125, 0.8921780986624299]

Training Information
-------------------------
input values: [5.  3.5 1.6 0.6]
expected values: [1. 0. 0.]
ouput values: [0.9859400660293207, 0.18601070656800997, 9.381167570654257e-12]

Training Information
-------------------------
input values: [5.7 3.  4.2 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0007612106770899056, 0.3407182925188372, 0.003596606533491853]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9865549842082941, 0.1873492301363994, 8.679879564410644e-12]

Training Information
-------------------------
input values: [5.5 3.5 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9872973669405306, 0.18627964493979202, 7.851797727875434e-12]

Training Information
-------------------------
input values: [6.2 2.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [1.685222638733031e-05, 0.40784255682369996, 0.7245954762321323]

Training Information
-------------------------
input values: [5.1 3.8 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.987331340261871, 0.1879981439743217, 7.703374838786466e-12]

Training Information
-------------------------
input values: [5.5 2.4 3.8 1.1]
expected values: [0. 1. 0.]
ouput values: [0.0019061900526061882, 0.33238500182723835, 0.0007221387921378117]

Training Information
-------------------------
input values: [5.4 3.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9870155549981552, 0.19010775349062897, 8.042309959578649e-12]

Training Information
-------------------------
input values: [5.8 2.7 3.9 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0040765599390932316, 0.3234977378579832, 0.00019336287757394833]

Training Information
-------------------------
input values: [7.7 3.8 6.7 2.2]
expected values: [0. 0. 1.]
ouput values: [9.806872510440685e-06, 0.4304854241312827, 0.8679019647261152]

Training Information
-------------------------
input values: [5.  3.5 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9871666277180228, 0.190182364163361, 7.894390517633304e-12]

Training Information
-------------------------
input values: [6.8 2.8 4.8 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00011741125394660125, 0.38252721994497757, 0.08255500309046748]

Training Information
-------------------------
input values: [7.  3.2 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0016851032347691638, 0.342119218002783, 0.0008949039038326779]

Training Information
-------------------------
input values: [5.4 3.9 1.7 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9871261931800254, 0.19432716145589293, 7.928898369937191e-12]

Training Information
-------------------------
input values: [5.  3.4 1.6 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9864025248682685, 0.19449102578458743, 8.730022350003889e-12]

Training Information
-------------------------
input values: [6.1 2.9 4.7 1.4]
expected values: [0. 1. 0.]
ouput values: [5.93006944149721e-05, 0.40219626756389376, 0.22658518928187224]

Training Information
-------------------------
input values: [6.5 3.  5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [1.4224980339486091e-05, 0.4326045973962834, 0.7747385236767821]

Training Information
-------------------------
input values: [5.8 4.  1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9875578717915496, 0.1935664073025548, 7.46920796609898e-12]

Training Information
-------------------------
input values: [5.  3.3 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9870574365572571, 0.19355888480161912, 8.00897285929912e-12]

Training Information
-------------------------
input values: [6.7 3.  5.  1.7]
expected values: [0. 1. 0.]
ouput values: [3.775715323028412e-05, 0.4098816806471632, 0.3900949141728474]

Training Information
-------------------------
input values: [7.4 2.8 6.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.4414816518745147e-05, 0.4320905244737761, 0.7693847737281756]

Training Information
-------------------------
input values: [5.8 2.8 5.1 2.4]
expected values: [0. 0. 1.]
ouput values: [1.00079186671511e-05, 0.4348709086298253, 0.8631521016476951]

Training Information
-------------------------
input values: [5.  3.  1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9868167646617029, 0.19220814348382043, 8.242939457992656e-12]

Training Information
-------------------------
input values: [6.4 3.1 5.5 1.8]
expected values: [0. 0. 1.]
ouput values: [2.345271146094236e-05, 0.41504512008175387, 0.5918556764066321]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [1.0501602960475862e-05, 0.42579347959265934, 0.8551098955205657]

Training Information
-------------------------
input values: [5.6 2.9 3.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.015224733191219628, 0.30076125215197635, 1.9658228653326427e-05]

Training Information
-------------------------
input values: [7.6 3.  6.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.307105722523194e-06, 0.43073126544721096, 0.8987336528762024]

Training Information
-------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.986723633580169, 0.18929966885232077, 8.452899645540365e-12]

Training Information
-------------------------
input values: [5.  2.  3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0002953218637137797, 0.36439702320410355, 0.018145091697433523]

Training Information
-------------------------
input values: [4.4 3.  1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9862156032161646, 0.19156591246918217, 9.0410726565334e-12]

Training Information
-------------------------
input values: [5.5 2.6 4.4 1.2]
expected values: [0. 1. 0.]
ouput values: [5.869907265216115e-05, 0.3959100962730017, 0.2320266668131181]

Training Information
-------------------------
input values: [5.7 2.9 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0008128379542224792, 0.35547970098712317, 0.003176660506999158]

Training Information
-------------------------
input values: [6.5 2.8 4.6 1.5]
expected values: [0. 1. 0.]
ouput values: [0.00010773435526793246, 0.39463084270243964, 0.09510053351676066]

Training Information
-------------------------
input values: [6.  2.2 4.  1. ]
expected values: [0. 1. 0.]
ouput values: [0.0008751492327730326, 0.36279755205893915, 0.0027936292621929177]

Training Information
-------------------------
input values: [6.2 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0008276046862820023, 0.36802678980497555, 0.003076148428881528]

Training Information
-------------------------
input values: [4.9 3.  1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9864816225539239, 0.20244293954610354, 8.682520851697344e-12]

Training Information
-------------------------
input values: [6.7 3.1 4.7 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0001742249867064636, 0.39906803207296454, 0.043717120954200446]

Training Information
-------------------------
input values: [6.2 3.4 5.4 2.3]
expected values: [0. 0. 1.]
ouput values: [9.718874740185247e-06, 0.4566024597181891, 0.8705566302839496]

Training Information
-------------------------
input values: [5.7 2.5 5.  2. ]
expected values: [0. 0. 1.]
ouput values: [8.923127044332317e-06, 0.45401089715719595, 0.8865336595010275]

Training Information
-------------------------
input values: [5.5 2.4 3.7 1. ]
expected values: [0. 1. 0.]
ouput values: [0.0014097699801569159, 0.35960147627467376, 0.0012297677986480141]

Training Information
-------------------------
input values: [6.7 3.  5.2 2.3]
expected values: [0. 0. 1.]
ouput values: [9.577143294175987e-06, 0.4534271354745849, 0.8738189555846916]

Training Information
-------------------------
input values: [4.5 2.3 1.3 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9818233046949711, 0.2047911699660664, 1.4696104916923193e-11]

Training Information
-------------------------
input values: [7.7 3.  6.1 2.3]
expected values: [0. 0. 1.]
ouput values: [8.418376482529445e-06, 0.45082611790199073, 0.8966277410690815]

Training Information
-------------------------
input values: [5.  3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9868285475499065, 0.19852256302793594, 8.332182559823865e-12]

Training Information
-------------------------
input values: [7.7 2.6 6.9 2.3]
expected values: [0. 0. 1.]
ouput values: [7.70842509190765e-06, 0.4475282999396438, 0.9100408577131793]

Training Information
-------------------------
input values: [6.7 3.1 4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.000656178050457122, 0.3649505439636806, 0.004624540989888192]

Training Information
-------------------------
input values: [5.8 2.7 5.1 1.9]
expected values: [0. 0. 1.]
ouput values: [9.13701782975024e-06, 0.4452805435638585, 0.8830071796609563]

Training Information
-------------------------
input values: [6.8 3.2 5.9 2.3]
expected values: [0. 0. 1.]
ouput values: [8.296281027024476e-06, 0.44298941399866065, 0.8993420720267908]

Training Information
-------------------------
input values: [4.8 3.1 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9857018297700901, 0.196089260966458, 9.653717721313986e-12]

Training Information
-------------------------
input values: [5.8 2.6 4.  1.2]
expected values: [0. 1. 0.]
ouput values: [0.00019849561493786417, 0.38207337433684224, 0.035569128616176324]

Training Information
-------------------------
input values: [5.6 2.8 4.9 2. ]
expected values: [0. 0. 1.]
ouput values: [9.248250093575778e-06, 0.44125187895571855, 0.8811454562175641]

Training Information
-------------------------
input values: [6.9 3.1 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [4.47515805646215e-05, 0.4091056957196542, 0.32697886179906266]

Training Information
-------------------------
input values: [5.1 3.7 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9871247697592542, 0.1972116446632532, 7.978900132161139e-12]

Training Information
-------------------------
input values: [6.6 2.9 4.6 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0006476058739667412, 0.3664451797823114, 0.004706819107073641]

Training Information
-------------------------
input values: [4.7 3.2 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.986364943209936, 0.19974770751325246, 8.829518831222987e-12]

Training Information
-------------------------
input values: [6.5 3.  5.8 2.2]
expected values: [0. 0. 1.]
ouput values: [8.681003527942334e-06, 0.44685676729396273, 0.8913209338670579]

Training Information
-------------------------
input values: [4.9 2.5 4.5 1.7]
expected values: [0. 0. 1.]
ouput values: [1.192508882274291e-05, 0.43703538213640436, 0.8259140142365076]

Training Information
-------------------------
input values: [6.4 3.2 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.0004452269977264354, 0.36953328270864433, 0.009006097154816203]

Training Information
-------------------------
input values: [4.4 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9865709615521103, 0.197930133902202, 8.636837322232371e-12]

Training Information
-------------------------
input values: [6.3 3.3 6.  2.5]
expected values: [0. 0. 1.]
ouput values: [8.056395989286003e-06, 0.44440405980416126, 0.9036752649418066]

Training Information
-------------------------
input values: [5.1 3.8 1.5 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9871930236583863, 0.1951249015203604, 7.940474374119816e-12]

Training Information
-------------------------
input values: [6.3 2.9 5.6 1.8]
expected values: [0. 0. 1.]
ouput values: [9.80385154343565e-06, 0.43605926788555227, 0.8699502213651158]

Training Information
-------------------------
input values: [4.6 3.1 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9861099606027565, 0.19391413287451567, 9.179250979933664e-12]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.1]
expected values: [0. 0. 1.]
ouput values: [8.631668014217046e-06, 0.43368237521009645, 0.8931156422286198]

Training Information
-------------------------
input values: [5.  3.6 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.987186660760512, 0.1908516696011818, 7.969755411508533e-12]

Training Information
-------------------------
input values: [4.6 3.2 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9865321074326666, 0.19097129004739183, 8.705098274691688e-12]

Training Information
-------------------------
input values: [6.1 2.6 5.6 1.4]
expected values: [0. 0. 1.]
ouput values: [1.036231966596792e-05, 0.42509171296737763, 0.8591814431149188]

Training Information
-------------------------
input values: [4.8 3.4 1.6 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9866327215319849, 0.18882457084348903, 8.600447400909294e-12]

Training Information
-------------------------
input values: [5.9 3.2 4.8 1.8]
expected values: [0. 1. 0.]
ouput values: [2.1032136364399896e-05, 0.40821435916422183, 0.642604971028481]

Training Information
-------------------------
input values: [6.7 3.3 5.7 2.1]
expected values: [0. 0. 1.]
ouput values: [1.4950602512176687e-05, 0.41922099496977233, 0.7605351344099288]

Training Information
-------------------------
input values: [5.6 2.5 3.9 1.1]
expected values: [0. 1. 0.]
ouput values: [0.003605889637766207, 0.32339700911499647, 0.00024101795004873795]

Training Information
-------------------------
input values: [5.3 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.987382556172746, 0.19063091139369237, 7.688628997532889e-12]

Training Information
-------------------------
input values: [5.1 3.3 1.7 0.5]
expected values: [1. 0. 0.]
ouput values: [0.9862116756545312, 0.19118714110781818, 8.983284169389689e-12]

Training Information
-------------------------
input values: [5.2 2.7 3.9 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00033128474844665507, 0.36522889278010057, 0.01482976948015423]

Training Information
-------------------------
input values: [6.  2.7 5.1 1.6]
expected values: [0. 1. 0.]
ouput values: [1.611720030356545e-05, 0.4223964603402212, 0.7373218598948148]

Training Information
-------------------------
input values: [6.4 2.8 5.6 2.2]
expected values: [0. 0. 1.]
ouput values: [1.0396289028825647e-05, 0.43530216915342235, 0.8543738036735075]

Training Information
-------------------------
input values: [6.3 2.8 5.1 1.5]
expected values: [0. 0. 1.]
ouput values: [6.786356840988722e-05, 0.3981315802212294, 0.1866890266629029]

Training Information
-------------------------
input values: [6.6 3.  4.4 1.4]
expected values: [0. 1. 0.]
ouput values: [0.00015513090621250493, 0.3806942268957187, 0.05292141629269893]

Training Information
-------------------------
input values: [4.3 3.  1.1 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9864159370041431, 0.19419756337092597, 8.776167907361842e-12]

Training Information
-------------------------
input values: [7.1 3.  5.9 2.1]
expected values: [0. 0. 1.]
ouput values: [8.35173775138762e-06, 0.43628164148982695, 0.897331877436463]

Training Information
-------------------------
input values: [6.8 3.  5.5 2.1]
expected values: [0. 0. 1.]
ouput values: [8.840922315905207e-06, 0.43132490989730315, 0.8880428722202257]

Training Information
-------------------------
input values: [5.5 2.5 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [4.366868705491039e-05, 0.3993627540766837, 0.33413316426975553]

Training Information
-------------------------
input values: [6.5 3.2 5.1 2. ]
expected values: [0. 0. 1.]
ouput values: [1.5451226437468936e-05, 0.4225819748913445, 0.7495981275793189]

Training Information
-------------------------
input values: [6.3 2.5 5.  1.9]
expected values: [0. 0. 1.]
ouput values: [9.913783731858859e-06, 0.4267895770268557, 0.8665417341571884]

Training Information
-------------------------
input values: [4.6 3.4 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9865440443940832, 0.18929948118671755, 8.61871224975418e-12]

Training Information
-------------------------
input values: [6.1 2.8 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [0.0007111433287112104, 0.34920692233300993, 0.004000156518146354]

Training Information
-------------------------
input values: [6.4 2.7 5.3 1.9]
expected values: [0. 0. 1.]
ouput values: [9.30572310296181e-06, 0.4283115306700383, 0.8789468703680374]

Training Information
-------------------------
input values: [6.7 2.5 5.8 1.8]
expected values: [0. 0. 1.]
ouput values: [8.376775676324403e-06, 0.42635581485521895, 0.8971802165474381]

Training Information
-------------------------
input values: [6.4 3.2 5.3 2.3]
expected values: [0. 0. 1.]
ouput values: [9.152051763060638e-06, 0.42099518243666284, 0.8823231237463911]

Training Information
-------------------------
input values: [6.9 3.2 5.7 2.3]
expected values: [0. 0. 1.]
ouput values: [8.675230918244323e-06, 0.4182095632539247, 0.8917716092832404]

Training Information
-------------------------
input values: [5.2 4.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9874219128387655, 0.1840639144323084, 7.685477795780264e-12]

Training Information
-------------------------
input values: [7.9 3.8 6.4 2. ]
expected values: [0. 0. 1.]
ouput values: [1.0782054160316456e-05, 0.41008923232564687, 0.8499926981457586]

Training Information
-------------------------
input values: [5.5 2.3 4.  1.3]
expected values: [0. 1. 0.]
ouput values: [3.551363631387063e-05, 0.3862706267334062, 0.4196583669937861]

Training Information
-------------------------
input values: [5.4 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [6.561240392389256e-05, 0.38073781972275733, 0.19778871425965627]

Training Information
-------------------------
input values: [6.2 2.8 4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [2.918908868123153e-05, 0.399368739031548, 0.49901924882827686]

Training Information
-------------------------
input values: [5.9 3.  4.2 1.5]
expected values: [0. 1. 0.]
ouput values: [0.000104351358561067, 0.37449963370329087, 0.10069105051404582]

Training Information
-------------------------
input values: [5.2 3.5 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9870088804668505, 0.18768288499934765, 8.135928601583868e-12]

Training Information
-------------------------
input values: [7.2 3.  5.8 1.6]
expected values: [0. 0. 1.]
ouput values: [1.173132765297962e-05, 0.4161385383864861, 0.8304913162366125]

Training Information
-------------------------
input values: [7.2 3.2 6.  1.8]
expected values: [0. 0. 1.]
ouput values: [9.880453221975832e-06, 0.4155104821492852, 0.8687100958726133]

Training Information
-------------------------
input values: [4.8 3.  1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9856715889370387, 0.1852632155458037, 9.706501219953239e-12]

Training Information
-------------------------
input values: [6.  3.4 4.5 1.6]
expected values: [0. 1. 0.]
ouput values: [8.673821028631476e-05, 0.3742286921923333, 0.13406303377902484]

Epoch 10000 root-mean-square error: 2.8968223757941574

Final root-mean-square error: 2.8968223757941574

Testing Information
------------------------
input values: [4.7 3.2 1.3 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9866661123011891, 0.1862754221160256, 8.54472756565413e-12]

Testing Information
------------------------
input values: [4.4 2.9 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9853509596738901, 0.18689713513801, 1.0088832196434716e-11]

Testing Information
------------------------
input values: [4.9 3.1 1.5 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9865143410231371, 0.18554029110096046, 8.713463577347251e-12]

Testing Information
------------------------
input values: [5.4 3.7 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9872488852707675, 0.18449565183782807, 7.892567191965057e-12]

Testing Information
------------------------
input values: [4.8 3.  1.4 0.1]
expected values: [1. 0. 0.]
ouput values: [0.9864552112519563, 0.18475044073996014, 8.784327654355863e-12]

Testing Information
------------------------
input values: [5.7 4.4 1.5 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9874750777324982, 0.18346389406136757, 7.647503226714351e-12]

Testing Information
------------------------
input values: [5.4 3.9 1.3 0.4]
expected values: [1. 0. 0.]
ouput values: [0.9873302507094738, 0.18317952665335302, 7.804822927642011e-12]

Testing Information
------------------------
input values: [5.1 3.5 1.4 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9869637827275348, 0.18308288752590604, 8.207476905731832e-12]

Testing Information
------------------------
input values: [5.7 3.8 1.7 0.3]
expected values: [1. 0. 0.]
ouput values: [0.9871706489432414, 0.1824960350174292, 7.977199620314042e-12]

Testing Information
------------------------
input values: [5.4 3.4 1.7 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9867750311617272, 0.18242165676216804, 8.415329842162417e-12]

Testing Information
------------------------
input values: [4.6 3.6 1.  0.2]
expected values: [1. 0. 0.]
ouput values: [0.9872987842578457, 0.18159086682545217, 7.850400731734801e-12]

Testing Information
------------------------
input values: [4.8 3.4 1.9 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9857587850479756, 0.18242133959399193, 9.586255975487134e-12]

Testing Information
------------------------
input values: [5.2 3.4 1.4 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9870456292948151, 0.18099065676146925, 8.11867599827815e-12]

Testing Information
------------------------
input values: [5.  3.2 1.2 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9869811364909019, 0.1806513016122021, 8.19375107615121e-12]

Testing Information
------------------------
input values: [5.1 3.4 1.5 0.2]
expected values: [1. 0. 0.]
ouput values: [0.9868837112670477, 0.18033231378335798, 8.298156023853349e-12]

Testing Information
------------------------
input values: [5.7 2.8 4.5 1.3]
expected values: [0. 1. 0.]
ouput values: [4.513822626001636e-05, 0.38025199929209264, 0.32357667546414737]

Testing Information
------------------------
input values: [6.3 3.3 4.7 1.6]
expected values: [0. 1. 0.]
ouput values: [0.00026127736809736365, 0.35552310909461066, 0.022222573646303916]

Testing Information
------------------------
input values: [4.9 2.4 3.3 1. ]
expected values: [0. 1. 0.]
ouput values: [0.008947064580777191, 0.3031192485994929, 4.9694158988189024e-05]

Testing Information
------------------------
input values: [5.6 3.  4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [6.0931330938224676e-05, 0.3889723933696573, 0.21990459998305179]

Testing Information
------------------------
input values: [5.8 2.7 4.1 1. ]
expected values: [0. 1. 0.]
ouput values: [0.008621406963781997, 0.31117067156418116, 5.2747050036688106e-05]

Testing Information
------------------------
input values: [6.3 2.5 4.9 1.5]
expected values: [0. 1. 0.]
ouput values: [2.2804957813323392e-05, 0.415559555359579, 0.6054082289406467]

Testing Information
------------------------
input values: [6.1 2.8 4.7 1.2]
expected values: [0. 1. 0.]
ouput values: [0.0030508880404260534, 0.3357078471973821, 0.00031499207926932125]

Testing Information
------------------------
input values: [6.4 2.9 4.3 1.3]
expected values: [0. 1. 0.]
ouput values: [0.04322425486497467, 0.29643643059095814, 3.0001414835535798e-06]

Testing Information
------------------------
input values: [6.  2.9 4.5 1.5]
expected values: [0. 1. 0.]
ouput values: [0.000500324460830741, 0.3744116435783426, 0.007153392730985152]

Testing Information
------------------------
input values: [5.7 2.6 3.5 1. ]
expected values: [0. 1. 0.]
ouput values: [0.29982647758653364, 0.26915874577106563, 6.14334628933133e-08]

Testing Information
------------------------
input values: [6.3 2.3 4.4 1.3]
expected values: [0. 1. 0.]
ouput values: [4.802968157331232e-05, 0.4246502834315158, 0.29043978011195365]

Testing Information
------------------------
input values: [5.6 3.  4.1 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0029474583781413507, 0.3557214805246557, 0.00032778568362836153]

Testing Information
------------------------
input values: [6.1 3.  4.6 1.4]
expected values: [0. 1. 0.]
ouput values: [0.0002413579079990854, 0.40440211758814326, 0.024301552259202046]

Testing Information
------------------------
input values: [5.  2.3 3.3 1. ]
expected values: [0. 1. 0.]
ouput values: [0.006704090423572826, 0.3495600998638484, 7.87173458915821e-05]

Testing Information
------------------------
input values: [5.6 2.7 4.2 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0001589246574837468, 0.4210618636538265, 0.0487891825081293]

Testing Information
------------------------
input values: [5.1 2.5 3.  1.1]
expected values: [0. 1. 0.]
ouput values: [0.05130460493562033, 0.3218252669663721, 2.156427866897588e-06]

Testing Information
------------------------
input values: [5.7 2.8 4.1 1.3]
expected values: [0. 1. 0.]
ouput values: [0.0002642185395371322, 0.42039459113942396, 0.020837748042959868]

Testing Information
------------------------
input values: [7.3 2.9 6.3 1.8]
expected values: [0. 0. 1.]
ouput values: [8.689683091159672e-06, 0.49006222904195257, 0.8862236603771931]

Testing Information
------------------------
input values: [7.2 3.6 6.1 2.5]
expected values: [0. 0. 1.]
ouput values: [8.544733303193646e-06, 0.48582131900516884, 0.8892911944414588]

Testing Information
------------------------
input values: [6.  2.2 5.  1.5]
expected values: [0. 0. 1.]
ouput values: [1.0351410951751517e-05, 0.47764873045427814, 0.8524099038494392]

Testing Information
------------------------
input values: [7.7 2.8 6.7 2. ]
expected values: [0. 0. 1.]
ouput values: [7.953146280439442e-06, 0.4782368906436009, 0.9013162812407651]

Testing Information
------------------------
input values: [6.3 2.7 4.9 1.8]
expected values: [0. 0. 1.]
ouput values: [1.2459759843288514e-05, 0.46533649297091195, 0.8080183191766057]

Testing Information
------------------------
input values: [6.1 3.  4.9 1.8]
expected values: [0. 0. 1.]
ouput values: [1.4085290468469865e-05, 0.45881636860684405, 0.7737422490738911]

Testing Information
------------------------
input values: [6.3 3.4 5.6 2.4]
expected values: [0. 0. 1.]
ouput values: [8.39867531634109e-06, 0.4642946490522623, 0.8937177752042719]

Testing Information
------------------------
input values: [6.  3.  4.8 1.8]
expected values: [0. 0. 1.]
ouput values: [1.3784994591366863e-05, 0.4508538482421069, 0.7814454192592359]

Testing Information
------------------------
input values: [6.9 3.1 5.4 2.1]
expected values: [0. 0. 1.]
ouput values: [9.484622973118037e-06, 0.4536764155847974, 0.8728132979535322]

Testing Information
------------------------
input values: [6.9 3.1 5.1 2.3]
expected values: [0. 0. 1.]
ouput values: [9.666742770270724e-06, 0.4491911558163259, 0.8693559868675321]

Testing Information
------------------------
input values: [6.7 3.3 5.7 2.5]
expected values: [0. 0. 1.]
ouput values: [8.092106080158334e-06, 0.44834121111063874, 0.9006909084617873]

Testing Information
------------------------
input values: [6.5 3.  5.2 2. ]
expected values: [0. 0. 1.]
ouput values: [9.695952382576565e-06, 0.4409803256865694, 0.8691643965724031]

Testing Information
------------------------
input values: [5.9 3.  5.1 1.8]
expected values: [0. 0. 1.]
ouput values: [1.0131986080748732e-05, 0.43621828421699493, 0.8605358011775568]

Final root-mean-square error: 1.9369953105102364

Sine Run
------------------------------------------------------------------

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.7215560141264195]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.705271513577573]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.7100207362228387]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.7200531366222762]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.7201845382319874]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.7005022154234251]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.7093268424224441]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.7213801165048705]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.7033872109042937]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.7249044749799587]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.709861967047276]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.7002595729895891]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.7042283208458555]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7121649785303651]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.7139929139343215]

Epoch 0 root-mean-square error: 1.2601480512245709

Epoch 100 root-mean-square error: 1.1917744965967783

Epoch 200 root-mean-square error: 1.1631643608241993

Epoch 300 root-mean-square error: 1.1214579167836225

Epoch 400 root-mean-square error: 1.050335519426507

Epoch 500 root-mean-square error: 0.9477151016359278

Epoch 600 root-mean-square error: 0.8377477654326226

Epoch 700 root-mean-square error: 0.7402894184308371

Epoch 800 root-mean-square error: 0.659816616288428

Epoch 900 root-mean-square error: 0.5942724023164055

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7138008049745761]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.44111650414897746]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.7905050184017307]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.7420450458452171]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.8303274932049836]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.8071051427956037]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.4905228590175266]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6233152530890278]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.6091759341837885]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.4108120646037489]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.28619369803553096]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.7873172612860955]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6132227294576738]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.3198154469587192]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.7994200820913686]

Epoch 1000 root-mean-square error: 0.540546518194947

Epoch 1100 root-mean-square error: 0.4959483014107479

Epoch 1200 root-mean-square error: 0.4584200849612702

Epoch 1300 root-mean-square error: 0.4264289973663208

Epoch 1400 root-mean-square error: 0.3988803039530567

Epoch 1500 root-mean-square error: 0.3749037827101996

Epoch 1600 root-mean-square error: 0.3538686988309737

Epoch 1700 root-mean-square error: 0.3352772027884061

Epoch 1800 root-mean-square error: 0.3187543403952754

Epoch 1900 root-mean-square error: 0.3039780088557836

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.8572263281822112]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.3572104575868584]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.7929187959584791]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6170196289945584]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.4285289648968735]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.8478677900600371]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.1985722836078317]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6305098933766221]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.31333593226899786]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.16132152772547736]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.757390864583694]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.8860192802308481]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.8638068404759965]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.6096818667829227]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.844435231612978]

Epoch 2000 root-mean-square error: 0.2907057393931731

Epoch 2100 root-mean-square error: 0.2787337258292921

Epoch 2200 root-mean-square error: 0.26789299002631617

Epoch 2300 root-mean-square error: 0.25804408431113746

Epoch 2400 root-mean-square error: 0.24907304098222996

Epoch 2500 root-mean-square error: 0.24087433624694743

Epoch 2600 root-mean-square error: 0.2333630704054965

Epoch 2700 root-mean-square error: 0.2264668500485642

Epoch 2800 root-mean-square error: 0.22012112819967744

Epoch 2900 root-mean-square error: 0.21426876760620148

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.3230540885181641]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7743071821541]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8125473496298085]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.2768938980555558]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.8871659915315111]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.12673451993518278]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.8673491008868114]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6308674904003412]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9094447389219297]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.16145901233309795]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.606880479145879]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.8803314325543558]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.40127357442193723]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.8708767968907687]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6150802830712708]

Epoch 3000 root-mean-square error: 0.2088652542279143

Epoch 3100 root-mean-square error: 0.20386291785591604

Epoch 3200 root-mean-square error: 0.19922616772936175

Epoch 3300 root-mean-square error: 0.1949210521555678

Epoch 3400 root-mean-square error: 0.1909176698781587

Epoch 3500 root-mean-square error: 0.18719035376062462

Epoch 3600 root-mean-square error: 0.18371256489528212

Epoch 3700 root-mean-square error: 0.1804669435078196

Epoch 3800 root-mean-square error: 0.1774303255024433

Epoch 3900 root-mean-square error: 0.17458948132808388

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.30553287650014005]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8231173344250896]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.258751602173714]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.3862057680524483]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.8930347170376779]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7829961937119455]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.11206544167392671]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.8999643097962674]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6128574813527311]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6295776461011162]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.8796965108402082]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.8833624898208723]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.14505702685596275]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9222390179761238]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.6041731361872743]

Epoch 4000 root-mean-square error: 0.17192531182993298

Epoch 4100 root-mean-square error: 0.16942398857195515

Epoch 4200 root-mean-square error: 0.16707426610749712

Epoch 4300 root-mean-square error: 0.16486286764279826

Epoch 4400 root-mean-square error: 0.16278045340506728

Epoch 4500 root-mean-square error: 0.16081713314329446

Epoch 4600 root-mean-square error: 0.15896333655729472

Epoch 4700 root-mean-square error: 0.15721115859481882

Epoch 4800 root-mean-square error: 0.15555267017536398

Epoch 4900 root-mean-square error: 0.15398367102951527

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2955041744669236]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.24868259009689958]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.628330195046852]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.13669352298037835]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8293805062056607]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7877372028835448]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.6014072162763382]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.8910399485587567]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.10478580884504354]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9302531353013614]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6104445497255742]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.887381910436074]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9009038802924244]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9079527673229131]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.3771087502740138]

Epoch 5000 root-mean-square error: 0.15249400066951768

Epoch 5100 root-mean-square error: 0.15108226996691504

Epoch 5200 root-mean-square error: 0.14974037199615306

Epoch 5300 root-mean-square error: 0.14846476668032946

Epoch 5400 root-mean-square error: 0.1472499383010763

Epoch 5500 root-mean-square error: 0.14609282216933844

Epoch 5600 root-mean-square error: 0.14498932569798711

Epoch 5700 root-mean-square error: 0.14393472309737831

Epoch 5800 root-mean-square error: 0.14292983267217219

Epoch 5900 root-mean-square error: 0.14196746850411565

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9064687181698552]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.5998194599423472]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9359712239843417]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6090060167739667]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.24269840953732755]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.8928184625631769]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6270250378638781]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.37123327928289]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.8965133691637341]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.913534196454201]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2895416636042325]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.10114220208722147]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.13230709414126973]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8337087150799282]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7910319989676411]

Epoch 6000 root-mean-square error: 0.14104707197967908

Epoch 6100 root-mean-square error: 0.14016475809720322

Epoch 6200 root-mean-square error: 0.13931719805390144

Epoch 6300 root-mean-square error: 0.1385084245494911

Epoch 6400 root-mean-square error: 0.13772906750504077

Epoch 6500 root-mean-square error: 0.13698091620499558

Epoch 6600 root-mean-square error: 0.13625985288648296

Epoch 6700 root-mean-square error: 0.1355688830708533

Epoch 6800 root-mean-square error: 0.1349013999257161

Epoch 6900 root-mean-square error: 0.1342586248569982

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.2391711341115075]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.0991588988177841]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.910395984633772]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.625887865278735]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8364017612047334]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.5979301097544708]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.36728717583688586]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2855051005522521]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9400790542093737]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.1298394443515036]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9175058900210193]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9004030897833899]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6076076869324344]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.792935874550269]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.8965749928038181]

Epoch 7000 root-mean-square error: 0.1336394225612647

Epoch 7100 root-mean-square error: 0.13304137319668372

Epoch 7200 root-mean-square error: 0.1324642437154944

Epoch 7300 root-mean-square error: 0.13190601632604082

Epoch 7400 root-mean-square error: 0.13136630129702587

Epoch 7500 root-mean-square error: 0.13084489251662235

Epoch 7600 root-mean-square error: 0.130339521799907

Epoch 7700 root-mean-square error: 0.1298504225654021

Epoch 7800 root-mean-square error: 0.1293761331988014

Epoch 7900 root-mean-square error: 0.1289161980394832

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9032161085266966]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.8994638752516338]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.23702990498159154]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2833054138996146]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.09822731464207084]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6063970425274354]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8385057932499944]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.5966828978003835]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.36476758156663464]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7942366766267507]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9432259646232467]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9205282383146308]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9133827660864677]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6248573964855508]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.12854033506754203]

Epoch 8000 root-mean-square error: 0.12847076055232962

Epoch 8100 root-mean-square error: 0.12803815480665573

Epoch 8200 root-mean-square error: 0.12761672127812201

Epoch 8300 root-mean-square error: 0.1272086127387879

Epoch 8400 root-mean-square error: 0.12681296526878283

Epoch 8500 root-mean-square error: 0.12642757298011262

Epoch 8600 root-mean-square error: 0.12605160583628358

Epoch 8700 root-mean-square error: 0.1256864088076283

Epoch 8800 root-mean-square error: 0.12533114838805173

Epoch 8900 root-mean-square error: 0.1249854242123095

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9054279351014326]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.5954665978622126]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.12784431964130827]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6048602489322576]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9456761330708133]

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2813633000231683]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.9015679674596767]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9228989916990602]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8399672878991317]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.23554789195535242]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.623820437107082]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7950998705271715]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.09770584587007819]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9155759342963743]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.36283436392701157]

Epoch 9000 root-mean-square error: 0.12464768718324541

Epoch 9100 root-mean-square error: 0.12431878544030098

Epoch 9200 root-mean-square error: 0.12399704507345122

Epoch 9300 root-mean-square error: 0.12368468499017819

Epoch 9400 root-mean-square error: 0.12337990967101244

Epoch 9500 root-mean-square error: 0.12308106942766972

Epoch 9600 root-mean-square error: 0.12279036747623681

Epoch 9700 root-mean-square error: 0.12250585103489814

Epoch 9800 root-mean-square error: 0.12222771714646356

Epoch 9900 root-mean-square error: 0.12195691054069331

Training Information
-------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.28030604356464867]

Training Information
-------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.127571426380805]

Training Information
-------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.0975933397015954]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.5943346836275376]

Training Information
-------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9477182789183015]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.9033084320439072]

Training Information
-------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9175070893833344]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.3615855439074982]

Training Information
-------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.623008859742453]

Training Information
-------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8410312950929707]

Training Information
-------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6039597376400644]

Training Information
-------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9071330526473346]

Training Information
-------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9247718381379636]

Training Information
-------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.795776891334209]

Training Information
-------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.2345103893186056]

Epoch 10000 root-mean-square error: 0.1216913201792942

Final root-mean-square error: 0.1216913201792942

Testing Information
------------------------
input values: [0.]
expected values: [0.]
ouput values: [0.07751351317202007]

Testing Information
------------------------
input values: [0.01]
expected values: [0.00999983]
ouput values: [0.08118453383605671]

Testing Information
------------------------
input values: [0.02]
expected values: [0.01999867]
ouput values: [0.0850134068597704]

Testing Information
------------------------
input values: [0.03]
expected values: [0.0299955]
ouput values: [0.08900490540540652]

Testing Information
------------------------
input values: [0.04]
expected values: [0.03998933]
ouput values: [0.09316375710465499]

Testing Information
------------------------
input values: [0.06]
expected values: [0.05996401]
ouput values: [0.10201726530415503]

Testing Information
------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.10670649007040475]

Testing Information
------------------------
input values: [0.08]
expected values: [0.07991469]
ouput values: [0.11158112586251735]

Testing Information
------------------------
input values: [0.09]
expected values: [0.08987855]
ouput values: [0.11664534458837221]

Testing Information
------------------------
input values: [0.1]
expected values: [0.09983342]
ouput values: [0.12190313172986202]

Testing Information
------------------------
input values: [0.12]
expected values: [0.11971221]
ouput values: [0.1330241570716053]

Testing Information
------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.13888472667162471]

Testing Information
------------------------
input values: [0.14]
expected values: [0.13954311]
ouput values: [0.14495237154069132]

Testing Information
------------------------
input values: [0.15]
expected values: [0.14943813]
ouput values: [0.15122969660710156]

Testing Information
------------------------
input values: [0.16]
expected values: [0.15931821]
ouput values: [0.15771893332574552]

Testing Information
------------------------
input values: [0.17]
expected values: [0.16918235]
ouput values: [0.16442190518662303]

Testing Information
------------------------
input values: [0.18]
expected values: [0.17902957]
ouput values: [0.17133999271056147]

Testing Information
------------------------
input values: [0.19]
expected values: [0.18885889]
ouput values: [0.17847409818168186]

Testing Information
------------------------
input values: [0.2]
expected values: [0.19866933]
ouput values: [0.18582461041344897]

Testing Information
------------------------
input values: [0.21]
expected values: [0.2084599]
ouput values: [0.19339136989643774]

Testing Information
------------------------
input values: [0.22]
expected values: [0.21822962]
ouput values: [0.20117363473036165]

Testing Information
------------------------
input values: [0.23]
expected values: [0.22797752]
ouput values: [0.20917004779903267]

Testing Information
------------------------
input values: [0.24]
expected values: [0.23770263]
ouput values: [0.2173786057029202]

Testing Information
------------------------
input values: [0.25]
expected values: [0.24740396]
ouput values: [0.2257966300175236]

Testing Information
------------------------
input values: [0.27]
expected values: [0.26673144]
ouput values: [0.24320000472576292]

Testing Information
------------------------
input values: [0.28]
expected values: [0.27635565]
ouput values: [0.25222160394764825]

Testing Information
------------------------
input values: [0.29]
expected values: [0.28595223]
ouput values: [0.26143475863166005]

Testing Information
------------------------
input values: [0.3]
expected values: [0.29552021]
ouput values: [0.270833116530535]

Testing Information
------------------------
input values: [0.32]
expected values: [0.31456656]
ouput values: [0.2900838498354787]

Testing Information
------------------------
input values: [0.33]
expected values: [0.32404303]
ouput values: [0.2999900503182017]

Testing Information
------------------------
input values: [0.34]
expected values: [0.33348709]
ouput values: [0.31004857726917034]

Testing Information
------------------------
input values: [0.35]
expected values: [0.34289781]
ouput values: [0.32024941359446457]

Testing Information
------------------------
input values: [0.36]
expected values: [0.35227423]
ouput values: [0.33058188808786976]

Testing Information
------------------------
input values: [0.37]
expected values: [0.36161543]
ouput values: [0.34103472137719315]

Testing Information
------------------------
input values: [0.38]
expected values: [0.37092047]
ouput values: [0.3515960798761134]

Testing Information
------------------------
input values: [0.4]
expected values: [0.38941834]
ouput values: [0.37291251435262107]

Testing Information
------------------------
input values: [0.41]
expected values: [0.39860933]
ouput values: [0.3837224607229377]

Testing Information
------------------------
input values: [0.42]
expected values: [0.40776045]
ouput values: [0.39458948964649826]

Testing Information
------------------------
input values: [0.43]
expected values: [0.4168708]
ouput values: [0.4055000479098151]

Testing Information
------------------------
input values: [0.44]
expected values: [0.42593947]
ouput values: [0.41644048034265035]

Testing Information
------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4273971167560607]

Testing Information
------------------------
input values: [0.46]
expected values: [0.44394811]
ouput values: [0.4383563585949775]

Testing Information
------------------------
input values: [0.47]
expected values: [0.45288629]
ouput values: [0.4493047638808227]

Testing Information
------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.4602291290714478]

Testing Information
------------------------
input values: [0.49]
expected values: [0.47062589]
ouput values: [0.4711165665605141]

Testing Information
------------------------
input values: [0.5]
expected values: [0.47942554]
ouput values: [0.48195457667161884]

Testing Information
------------------------
input values: [0.51]
expected values: [0.48817725]
ouput values: [0.49273111316738205]

Testing Information
------------------------
input values: [0.52]
expected values: [0.49688014]
ouput values: [0.5034346414824878]

Testing Information
------------------------
input values: [0.53]
expected values: [0.50553334]
ouput values: [0.5140541890935746]

Testing Information
------------------------
input values: [0.54]
expected values: [0.51413599]
ouput values: [0.5245793876489655]

Testing Information
------------------------
input values: [0.55]
expected values: [0.52268723]
ouput values: [0.5350005066889387]

Testing Information
------------------------
input values: [0.56]
expected values: [0.5311862]
ouput values: [0.5453084789846993]

Testing Information
------------------------
input values: [0.57]
expected values: [0.53963205]
ouput values: [0.555494917704893]

Testing Information
------------------------
input values: [0.58]
expected values: [0.54802394]
ouput values: [0.5655521257771421]

Testing Information
------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5754730979451186]

Testing Information
------------------------
input values: [0.6]
expected values: [0.56464247]
ouput values: [0.5852515161270727]

Testing Information
------------------------
input values: [0.63]
expected values: [0.58914476]
ouput values: [0.6140014482323597]

Testing Information
------------------------
input values: [0.65]
expected values: [0.60518641]
ouput values: [0.6323296851148168]

Testing Information
------------------------
input values: [0.66]
expected values: [0.61311685]
ouput values: [0.641151011717443]

Testing Information
------------------------
input values: [0.67]
expected values: [0.62098599]
ouput values: [0.6498030727126385]

Testing Information
------------------------
input values: [0.68]
expected values: [0.62879302]
ouput values: [0.6582843419630993]

Testing Information
------------------------
input values: [0.69]
expected values: [0.63653718]
ouput values: [0.666593754715299]

Testing Information
------------------------
input values: [0.7]
expected values: [0.64421769]
ouput values: [0.6747306746301274]

Testing Information
------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6826948611862504]

Testing Information
------------------------
input values: [0.72]
expected values: [0.65938467]
ouput values: [0.6904864377926591]

Testing Information
------------------------
input values: [0.73]
expected values: [0.66686964]
ouput values: [0.6981058608845542]

Testing Information
------------------------
input values: [0.74]
expected values: [0.67428791]
ouput values: [0.7055538902186761]

Testing Information
------------------------
input values: [0.75]
expected values: [0.68163876]
ouput values: [0.7128315605311897]

Testing Information
------------------------
input values: [0.76]
expected values: [0.68892145]
ouput values: [0.7199401546737944]

Testing Information
------------------------
input values: [0.77]
expected values: [0.69613524]
ouput values: [0.7268811783020303]

Testing Information
------------------------
input values: [0.78]
expected values: [0.70327942]
ouput values: [0.7336563361538174]

Testing Information
------------------------
input values: [0.79]
expected values: [0.71035327]
ouput values: [0.7402675099258882]

Testing Information
------------------------
input values: [0.8]
expected values: [0.71735609]
ouput values: [0.7467167377306527]

Testing Information
------------------------
input values: [0.81]
expected values: [0.72428717]
ouput values: [0.7530061950957799]

Testing Information
------------------------
input values: [0.82]
expected values: [0.73114583]
ouput values: [0.7591381774529223]

Testing Information
------------------------
input values: [0.83]
expected values: [0.73793137]
ouput values: [0.7651150840500762]

Testing Information
------------------------
input values: [0.84]
expected values: [0.74464312]
ouput values: [0.7709394032135647]

Testing Information
------------------------
input values: [0.85]
expected values: [0.75128041]
ouput values: [0.7766136988800869]

Testing Information
------------------------
input values: [0.86]
expected values: [0.75784256]
ouput values: [0.7821405983162196]

Testing Information
------------------------
input values: [0.87]
expected values: [0.76432894]
ouput values: [0.7875227809417787]

Testing Information
------------------------
input values: [0.89]
expected values: [0.77707175]
ouput values: [0.7979556539292995]

Testing Information
------------------------
input values: [0.9]
expected values: [0.78332691]
ouput values: [0.8029183115066434]

Testing Information
------------------------
input values: [0.91]
expected values: [0.78950374]
ouput values: [0.8077473240715786]

Testing Information
------------------------
input values: [0.92]
expected values: [0.79560162]
ouput values: [0.8124454981903739]

Testing Information
------------------------
input values: [0.93]
expected values: [0.80161994]
ouput values: [0.817015644134458]

Testing Information
------------------------
input values: [0.94]
expected values: [0.8075581]
ouput values: [0.8214605702905554]

Testing Information
------------------------
input values: [0.95]
expected values: [0.8134155]
ouput values: [0.8257830781086353]

Testing Information
------------------------
input values: [0.96]
expected values: [0.81919157]
ouput values: [0.8299859575372976]

Testing Information
------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8340719829008955]

Testing Information
------------------------
input values: [0.99]
expected values: [0.83602598]
ouput values: [0.841926691908065]

Testing Information
------------------------
input values: [1.]
expected values: [0.84147098]
ouput values: [0.8456781237966209]

Testing Information
------------------------
input values: [1.01]
expected values: [0.84683184]
ouput values: [0.8493235818436538]

Testing Information
------------------------
input values: [1.02]
expected values: [0.85210802]
ouput values: [0.8528657132604548]

Testing Information
------------------------
input values: [1.03]
expected values: [0.85729899]
ouput values: [0.8563071321212888]

Testing Information
------------------------
input values: [1.04]
expected values: [0.86240423]
ouput values: [0.8596504173915026]

Testing Information
------------------------
input values: [1.05]
expected values: [0.86742323]
ouput values: [0.8628981111672619]

Testing Information
------------------------
input values: [1.06]
expected values: [0.87235548]
ouput values: [0.8660527171106012]

Testing Information
------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8691166990653028]

Testing Information
------------------------
input values: [1.08]
expected values: [0.88195781]
ouput values: [0.8720924798406825]

Testing Information
------------------------
input values: [1.09]
expected values: [0.88662691]
ouput values: [0.8749824401517375]

Testing Information
------------------------
input values: [1.1]
expected values: [0.89120736]
ouput values: [0.8777889177052653]

Testing Information
------------------------
input values: [1.11]
expected values: [0.89569869]
ouput values: [0.8805142064225655]

Testing Information
------------------------
input values: [1.12]
expected values: [0.90010044]
ouput values: [0.883160555790173]

Testing Information
------------------------
input values: [1.13]
expected values: [0.90441219]
ouput values: [0.8857301703307777]

Testing Information
------------------------
input values: [1.14]
expected values: [0.9086335]
ouput values: [0.8882252091870854]

Testing Information
------------------------
input values: [1.15]
expected values: [0.91276394]
ouput values: [0.8906477858118603]

Testing Information
------------------------
input values: [1.16]
expected values: [0.91680311]
ouput values: [0.8929999677578149]

Testing Information
------------------------
input values: [1.17]
expected values: [0.9207506]
ouput values: [0.8952837765613498]

Testing Information
------------------------
input values: [1.18]
expected values: [0.92460601]
ouput values: [0.8975011877144394]

Testing Information
------------------------
input values: [1.19]
expected values: [0.92836897]
ouput values: [0.8996541307192105]

Testing Information
------------------------
input values: [1.21]
expected values: [0.935616]
ouput values: [0.9037357617899785]

Testing Information
------------------------
input values: [1.23]
expected values: [0.9424888]
ouput values: [0.9075822953839979]

Testing Information
------------------------
input values: [1.24]
expected values: [0.945784]
ouput values: [0.9094418101553101]

Testing Information
------------------------
input values: [1.25]
expected values: [0.94898462]
ouput values: [0.9112474417478044]

Testing Information
------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.9130008058834522]

Testing Information
------------------------
input values: [1.27]
expected values: [0.95510086]
ouput values: [0.9147034751186219]

Testing Information
------------------------
input values: [1.29]
expected values: [0.96083506]
ouput values: [0.9179239196347995]

Testing Information
------------------------
input values: [1.3]
expected values: [0.96355819]
ouput values: [0.919484295279242]

Testing Information
------------------------
input values: [1.31]
expected values: [0.96618495]
ouput values: [0.9209998270869403]

Testing Information
------------------------
input values: [1.32]
expected values: [0.9687151]
ouput values: [0.922471881608057]

Testing Information
------------------------
input values: [1.34]
expected values: [0.97348454]
ouput values: [0.9252541830762736]

Testing Information
------------------------
input values: [1.35]
expected values: [0.97572336]
ouput values: [0.9266043279459598]

Testing Information
------------------------
input values: [1.36]
expected values: [0.9778646]
ouput values: [0.9279160590891128]

Testing Information
------------------------
input values: [1.37]
expected values: [0.97990806]
ouput values: [0.9291905551270027]

Testing Information
------------------------
input values: [1.38]
expected values: [0.98185353]
ouput values: [0.9304289593612688]

Testing Information
------------------------
input values: [1.39]
expected values: [0.98370081]
ouput values: [0.9316323805794765]

Testing Information
------------------------
input values: [1.4]
expected values: [0.98544973]
ouput values: [0.9328018938646627]

Testing Information
------------------------
input values: [1.41]
expected values: [0.9871001]
ouput values: [0.9339385414067636]

Testing Information
------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9350433333139601]

Testing Information
------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9361172484221318]

Testing Information
------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9371612351007416]

Testing Information
------------------------
input values: [1.45]
expected values: [0.99271299]
ouput values: [0.9381762120536176]

Testing Information
------------------------
input values: [1.46]
expected values: [0.99386836]
ouput values: [0.9391630691132334]

Testing Information
------------------------
input values: [1.47]
expected values: [0.99492435]
ouput values: [0.9401226680272052]

Testing Information
------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9410558432358581]

Testing Information
------------------------
input values: [1.49]
expected values: [0.99673775]
ouput values: [0.9419634026398244]

Testing Information
------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9428461283567482]

Testing Information
------------------------
input values: [1.51]
expected values: [0.99815247]
ouput values: [0.9437047774662771]

Testing Information
------------------------
input values: [1.52]
expected values: [0.99871014]
ouput values: [0.9445400827426221]

Testing Information
------------------------
input values: [1.53]
expected values: [0.99916795]
ouput values: [0.9453527533740582]

Testing Information
------------------------
input values: [1.54]
expected values: [0.99952583]
ouput values: [0.9461434756688266]

Testing Information
------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9469129137469866]

Testing Information
------------------------
input values: [1.57]
expected values: [0.99999968]
ouput values: [0.9483718631537853]

Final root-mean-square error: 0.3822284310703703

XOR Run
------------------------------------------------------------------

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.7442244869729207]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.8068415952638759]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.7291253083639696]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.7944138785113127]

Epoch 0 root-mean-square error: 1.135915420072215

Epoch 100 root-mean-square error: 1.0204098859840915

Epoch 200 root-mean-square error: 1.0029502245929969

Epoch 300 root-mean-square error: 1.0017000740482138

Epoch 400 root-mean-square error: 1.001594977321258

Epoch 500 root-mean-square error: 1.0015643704920598

Epoch 600 root-mean-square error: 1.0015379073269983

Epoch 700 root-mean-square error: 1.001504352784881

Epoch 800 root-mean-square error: 1.001479377225097

Epoch 900 root-mean-square error: 1.0014564378147304

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.5011193038070535]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.4996644069720824]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.503940320047943]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.4996430467017492]

Epoch 1000 root-mean-square error: 1.0014286771970597

Epoch 1100 root-mean-square error: 1.001400812926899

Epoch 1200 root-mean-square error: 1.001370231356768

Epoch 1300 root-mean-square error: 1.001333472762921

Epoch 1400 root-mean-square error: 1.0013077968046087

Epoch 1500 root-mean-square error: 1.0012687039989354

Epoch 1600 root-mean-square error: 1.0012380690821538

Epoch 1700 root-mean-square error: 1.001200151578669

Epoch 1800 root-mean-square error: 1.0011548848953806

Epoch 1900 root-mean-square error: 1.0011170789745125

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.49983932750751764]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5075573031809035]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.4962075573875707]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.5018559958935914]

Epoch 2000 root-mean-square error: 1.0010716773319024

Epoch 2100 root-mean-square error: 1.001022480420101

Epoch 2200 root-mean-square error: 1.0009628714242758

Epoch 2300 root-mean-square error: 1.0009127670252216

Epoch 2400 root-mean-square error: 1.0008444775994414

Epoch 2500 root-mean-square error: 1.0007827964184364

Epoch 2600 root-mean-square error: 1.0007012511309699

Epoch 2700 root-mean-square error: 1.0006201181723808

Epoch 2800 root-mean-square error: 1.000536900125505

Epoch 2900 root-mean-square error: 1.000438608608645

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5088705625651134]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.49654930932469526]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.5069026021527412]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.49508071952945965]

Epoch 3000 root-mean-square error: 1.0003308502692105

Epoch 3100 root-mean-square error: 1.0002055790787743

Epoch 3200 root-mean-square error: 1.0000706952786127

Epoch 3300 root-mean-square error: 0.9999214298449316

Epoch 3400 root-mean-square error: 0.9997615791911512

Epoch 3500 root-mean-square error: 0.9995747792239846

Epoch 3600 root-mean-square error: 0.9993592891847869

Epoch 3700 root-mean-square error: 0.9991236404876331

Epoch 3800 root-mean-square error: 0.9988588839982121

Epoch 3900 root-mean-square error: 0.9985653352714583

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.48908141360337914]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5130095410127816]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.49362914549421205]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.5125113848081839]

Epoch 4000 root-mean-square error: 0.998216414680194

Epoch 4100 root-mean-square error: 0.9978368453397413

Epoch 4200 root-mean-square error: 0.9973980107051078

Epoch 4300 root-mean-square error: 0.9968888144131491

Epoch 4400 root-mean-square error: 0.9963183098756561

Epoch 4500 root-mean-square error: 0.9956743676982482

Epoch 4600 root-mean-square error: 0.9949299271657733

Epoch 4700 root-mean-square error: 0.9940744920508849

Epoch 4800 root-mean-square error: 0.9931078509354206

Epoch 4900 root-mean-square error: 0.9920146243470886

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.4976704236123299]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5274078165185355]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.4782093570348992]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.5283643271154446]

Epoch 5000 root-mean-square error: 0.9907665745934632

Epoch 5100 root-mean-square error: 0.9893507208354548

Epoch 5200 root-mean-square error: 0.9877516690353193

Epoch 5300 root-mean-square error: 0.9859452500415277

Epoch 5400 root-mean-square error: 0.983944584316645

Epoch 5500 root-mean-square error: 0.9816947089863879

Epoch 5600 root-mean-square error: 0.9792250579359537

Epoch 5700 root-mean-square error: 0.9764989589677353

Epoch 5800 root-mean-square error: 0.9735214630599992

Epoch 5900 root-mean-square error: 0.9702926532353986

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.45584197682946587]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.5629752455183855]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5347161039926815]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.4999721026198366]

Epoch 6000 root-mean-square error: 0.9668152635829919

Epoch 6100 root-mean-square error: 0.9630909019036783

Epoch 6200 root-mean-square error: 0.9591247539612202

Epoch 6300 root-mean-square error: 0.954950444758527

Epoch 6400 root-mean-square error: 0.9505748779027866

Epoch 6500 root-mean-square error: 0.9459986843540152

Epoch 6600 root-mean-square error: 0.9412496770349075

Epoch 6700 root-mean-square error: 0.9363387076961351

Epoch 6800 root-mean-square error: 0.9312876639638716

Epoch 6900 root-mean-square error: 0.9261258974423054

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.5163530658868671]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.5283723004046785]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.4216320184382181]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.6036394831756152]

Epoch 7000 root-mean-square error: 0.9208511622306529

Epoch 7100 root-mean-square error: 0.9154580210378446

Epoch 7200 root-mean-square error: 0.9099986046524271

Epoch 7300 root-mean-square error: 0.9044531456851642

Epoch 7400 root-mean-square error: 0.8988473894073676

Epoch 7500 root-mean-square error: 0.8931668873144881

Epoch 7600 root-mean-square error: 0.8874379807231854

Epoch 7700 root-mean-square error: 0.8816585660203571

Epoch 7800 root-mean-square error: 0.875836621673807

Epoch 7900 root-mean-square error: 0.8699825673944969

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.5519211902938049]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.506947901002269]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.6251894683898706]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.38523418982470015]

Epoch 8000 root-mean-square error: 0.8640943870053132

Epoch 8100 root-mean-square error: 0.8581731919218736

Epoch 8200 root-mean-square error: 0.8522696307688215

Epoch 8300 root-mean-square error: 0.8463264755463454

Epoch 8400 root-mean-square error: 0.8404029472163689

Epoch 8500 root-mean-square error: 0.8344994482097523

Epoch 8600 root-mean-square error: 0.8285939057424206

Epoch 8700 root-mean-square error: 0.8226951688963346

Epoch 8800 root-mean-square error: 0.8168024010774042

Epoch 8900 root-mean-square error: 0.8109411344346865

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.602889536884353]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.3468989334025238]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.6288341389763373]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.4819786680329591]

Epoch 9000 root-mean-square error: 0.8050485219786411

Epoch 9100 root-mean-square error: 0.7991533773910826

Epoch 9200 root-mean-square error: 0.7932242392148171

Epoch 9300 root-mean-square error: 0.787218023231136

Epoch 9400 root-mean-square error: 0.7811468080474073

Epoch 9500 root-mean-square error: 0.7749683408268454

Epoch 9600 root-mean-square error: 0.7686277722977445

Epoch 9700 root-mean-square error: 0.7620727500481193

Epoch 9800 root-mean-square error: 0.7552915480733416

Epoch 9900 root-mean-square error: 0.7481547336456902

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
ouput values: [0.6440033739532645]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
ouput values: [0.44314986424582853]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
ouput values: [0.31033429211862545]

Training Information
-------------------------
input values: [1. 0.]
expected values: [1.]
ouput values: [0.640635546940832]

Epoch 10000 root-mean-square error: 0.7406521335451879

Final root-mean-square error: 0.7406521335451879


Process finished with exit code 0

"""