"""
Assignment Five Check-In
   Jonathan Fong
   7/31/2020
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
                              f'ouput values: {output_nodes_value}\n')

                # finds the root-mean-square value of one training epoch
                rmse_value = math.sqrt(sum(training_errors) / self._output_nodes_count)
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
            rmse_value = math.sqrt(sum(testing_errors) / self._output_nodes_count)
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
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)

    xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_labels = [[0], [1], [1], [0]]

    data = NNData(xor_features, xor_labels, 1)
    network.train(data, 100000, order=NNData.Order.RANDOM)


def main():
    """Runs program"""
    print('Iris Run\n')
    print('------------------------------------------------------------------')
    run_iris()
    print('Sine Run\n')
    print('------------------------------------------------------------------\n')
    run_sin()
    print('XOR Run')
    print('------------------------------------------------------------------')
    run_XOR()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/AssignmentFiveCheck1.py"
Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.6226789289073379]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.6066733476005898]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.6077673482548857]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.6519846672173804]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.6374808498792418]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.6530647101404545]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.6527885828464177]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.6227457732548057]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.6533179403538959]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.6285465904277249]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.6501078771478906]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.633693859395162]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.6563243096422136]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.6587952317634924]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.6484059563858535]

Epoch 0 root-mean-square error: 1.2057407237786315

Epoch 100 root-mean-square error: 1.1114308765685315

Epoch 200 root-mean-square error: 1.104659434900334

Epoch 300 root-mean-square error: 1.1001043931198597

Epoch 400 root-mean-square error: 1.0968718526156223

Epoch 500 root-mean-square error: 1.0945129199687624

Epoch 600 root-mean-square error: 1.0927472053627807

Epoch 700 root-mean-square error: 1.091385404612955

Epoch 800 root-mean-square error: 1.0903201226977433

Epoch 900 root-mean-square error: 1.089474233392654

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.6765684596731945]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.7788648458339593]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.7597344334836524]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.7794795568560074]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.7660299227101486]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.7433598105209257]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.7746229135528194]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.7845782832198798]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.724019845846568]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.7831727790261718]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.7209915539052668]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.7334749105171489]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.6839698652859613]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.7804734282881959]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.7800207289737662]

Epoch 1000 root-mean-square error: 1.0887778740141216

Epoch 1100 root-mean-square error: 1.08818433346059

Epoch 1200 root-mean-square error: 1.0876579471202297

Epoch 1300 root-mean-square error: 1.087162687850672

Epoch 1400 root-mean-square error: 1.0867137296708915

Epoch 1500 root-mean-square error: 1.0862376628372048

Epoch 1600 root-mean-square error: 1.0856688333159732

Epoch 1700 root-mean-square error: 1.0850982544571797

Epoch 1800 root-mean-square error: 1.084330261882643

Epoch 1900 root-mean-square error: 1.0833150952650052

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.7800431871293139]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.780286864346751]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.7238190611603796]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.7197631526289627]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.7672664769458133]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.7625250797999766]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.7826888279413219]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.7760845915770407]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.735423464906804]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.7456660164142033]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.7817724539137697]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.781365647237894]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.6792124261928713]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.6691965510166453]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.7805778912266548]

Epoch 2000 root-mean-square error: 1.0818715096479827

Epoch 2100 root-mean-square error: 1.0796967003647246

Epoch 2200 root-mean-square error: 1.0761423242502395

Epoch 2300 root-mean-square error: 1.0698444361047281

Epoch 2400 root-mean-square error: 1.057497361945098

Epoch 2500 root-mean-square error: 1.0314145372950188

Epoch 2600 root-mean-square error: 0.9788733917558531

Epoch 2700 root-mean-square error: 0.8980417772866729

Epoch 2800 root-mean-square error: 0.8096231326472266

Epoch 2900 root-mean-square error: 0.7304221213748299

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.8138236467365487]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.6527371047102609]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.8327682350015829]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.7653656268101191]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.4026081028684447]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.5932887097960214]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.8367825528832311]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.842823122082537]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.8390643471608543]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.6072597644362159]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6950483327377897]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.8335536989229833]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.4332629809905497]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.7841665395038219]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.8315578815897248]

Epoch 3000 root-mean-square error: 0.6640596518981864

Epoch 3100 root-mean-square error: 0.6089606475743247

Epoch 3200 root-mean-square error: 0.5629449753337652

Epoch 3300 root-mean-square error: 0.5239887654591633

Epoch 3400 root-mean-square error: 0.49060485244871027

Epoch 3500 root-mean-square error: 0.46166970161506776

Epoch 3600 root-mean-square error: 0.4363261130533233

Epoch 3700 root-mean-square error: 0.41391841347100755

Epoch 3800 root-mean-square error: 0.3939814637249933

Epoch 3900 root-mean-square error: 0.37609390877095594

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.5389621734848811]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.21098873676429142]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.8916342243474991]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.8844183439583366]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.8864860275291312]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.2519913136143186]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8294559851319245]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.6206417330283412]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8027297397324269]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.5144217819127127]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.8647726686777606]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6944900552783634]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.8853142794227715]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.8900459794950567]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.895781252693659]

Epoch 4000 root-mean-square error: 0.35995963662113

Epoch 4100 root-mean-square error: 0.3453344120316112

Epoch 4200 root-mean-square error: 0.3320128238829982

Epoch 4300 root-mean-square error: 0.319829904215216

Epoch 4400 root-mean-square error: 0.3086523961457102

Epoch 4500 root-mean-square error: 0.2983585645065813

Epoch 4600 root-mean-square error: 0.2888503295867047

Epoch 4700 root-mean-square error: 0.2800462127270799

Epoch 4800 root-mean-square error: 0.2718713656929182

Epoch 4900 root-mean-square error: 0.2642638682063219

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.5082962580377002]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.6050145881129989]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9126045993755667]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9079681678544196]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9144442019091263]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9184486570411827]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8487027168724885]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4801637059970383]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.8868707611123482]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.15750029231896587]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9090902244225739]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8191560608058113]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9071675928786146]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.19615806969176755]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6937340553209829]

Epoch 5000 root-mean-square error: 0.2571733514682246

Epoch 5100 root-mean-square error: 0.25054439240693355

Epoch 5200 root-mean-square error: 0.24434436021148448

Epoch 5300 root-mean-square error: 0.23852858481718003

Epoch 5400 root-mean-square error: 0.23306718124941153

Epoch 5500 root-mean-square error: 0.22793042608393937

Epoch 5600 root-mean-square error: 0.22309116907436963

Epoch 5700 root-mean-square error: 0.21853024738549984

Epoch 5800 root-mean-square error: 0.21422333458389994

Epoch 5900 root-mean-square error: 0.21015145138343

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8598110308501682]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.1696315619630224]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9197309745047986]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4603229928864979]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.8993282182797527]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8286684401322453]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.13321829887190478]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9254135962987659]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.921771678227286]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6931048363248816]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.49074537173859634]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9309873853019337]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5961918359031396]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9270483489223722]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9207114691207813]

Epoch 6000 root-mean-square error: 0.20629825071083022

Epoch 6100 root-mean-square error: 0.20264746877160977

Epoch 6200 root-mean-square error: 0.19918490451074092

Epoch 6300 root-mean-square error: 0.19589982149725274

Epoch 6400 root-mean-square error: 0.19277806335734818

Epoch 6500 root-mean-square error: 0.18980976822309903

Epoch 6600 root-mean-square error: 0.18698531039137972

Epoch 6700 root-mean-square error: 0.18429401745241253

Epoch 6800 root-mean-square error: 0.18173027876629053

Epoch 6900 root-mean-square error: 0.1792843427643274

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9390521713061808]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.9075488981718302]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9289707447722845]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6929114683013735]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8673451284194174]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8351631936210375]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9352306939184911]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.1546668529195639]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9279493518693664]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5906781281677308]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.11979043843950532]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9298738275649702]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.4795715128354437]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9335221192244094]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4477339326577954]

Epoch 7000 root-mean-square error: 0.17695002899191822

Epoch 7100 root-mean-square error: 0.17471940102412828

Epoch 7200 root-mean-square error: 0.1725871852875182

Epoch 7300 root-mean-square error: 0.1705486353165039

Epoch 7400 root-mean-square error: 0.16859703782243723

Epoch 7500 root-mean-square error: 0.1667277261818134

Epoch 7600 root-mean-square error: 0.1649376048265021

Epoch 7700 root-mean-square error: 0.16322151515577477

Epoch 7800 root-mean-square error: 0.16157434143219318

Epoch 7900 root-mean-square error: 0.159993817213256

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.43910529586091956]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.47177675566361776]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9408297656409977]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9336299002271937]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9392187715509192]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5866113244292194]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8397479817302047]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.913261184074978]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9346316237567296]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.11161028905564717]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.1452862618445916]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6925174157966769]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9445955519243849]

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9355565137802901]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8726424946954612]

Epoch 8000 root-mean-square error: 0.15847650651345055

Epoch 8100 root-mean-square error: 0.15701802314968352

Epoch 8200 root-mean-square error: 0.15561597119457446

Epoch 8300 root-mean-square error: 0.15426751272007205

Epoch 8400 root-mean-square error: 0.15297028350838024

Epoch 8500 root-mean-square error: 0.15172084272797504

Epoch 8600 root-mean-square error: 0.15051787724928867

Epoch 8700 root-mean-square error: 0.14935799764159194

Epoch 8800 root-mean-square error: 0.14824011267869838

Epoch 8900 root-mean-square error: 0.14716217315752222

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.939812543260652]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9434070988622755]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8433464783570594]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9378991669663156]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.10634343467005627]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.13920345159389333]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8767154465173013]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6924999023848972]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9449983249185215]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9487593212377481]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5835119775255763]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4328801717091457]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.91753318591487]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9388422274766143]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.46632651497995176]

Epoch 9000 root-mean-square error: 0.1461217689254104

Epoch 9100 root-mean-square error: 0.14511736258559102

Epoch 9200 root-mean-square error: 0.1441472915346304

Epoch 9300 root-mean-square error: 0.14320987250831638

Epoch 9400 root-mean-square error: 0.1423041704402143

Epoch 9500 root-mean-square error: 0.14142709951062787

Epoch 9600 root-mean-square error: 0.14058099041575173

Epoch 9700 root-mean-square error: 0.13976036011859747

Epoch 9800 root-mean-square error: 0.13896694995342848

Epoch 9900 root-mean-square error: 0.1381975685907625

Training Information
-------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9430478731026016]

Training Information
-------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9421190936988804]

Training Information
-------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.10283253664981337]

Training Information
-------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.920916222481213]

Training Information
-------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9482711034514213]

Training Information
-------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4287647555137877]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
ouput values: [0.8460542157350952]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
ouput values: [0.9466488671891945]

Training Information
-------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6924875262060067]

Training Information
-------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.1350589401801771]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
ouput values: [0.4621956099593752]

Training Information
-------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9411275892856967]

Training Information
-------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9519521086882297]

Training Information
-------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8797728547364495]

Training Information
-------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5813576863475801]

Epoch 10000 root-mean-square error: 0.1374527702704337

Final root-mean-square error: 0.1374527702704337

Testing Information
------------------------
input values: [0.]
expected values: [0.]
ouput values: [0.07381677340412701]

Testing Information
------------------------
input values: [0.01]
expected values: [0.00999983]
ouput values: [0.07743147597369376]

Testing Information
------------------------
input values: [0.02]
expected values: [0.01999867]
ouput values: [0.08120767968034287]

Testing Information
------------------------
input values: [0.03]
expected values: [0.0299955]
ouput values: [0.08515059050619932]

Testing Information
------------------------
input values: [0.04]
expected values: [0.03998933]
ouput values: [0.08926538665642725]

Testing Information
------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.09355719744229232]

Testing Information
------------------------
input values: [0.06]
expected values: [0.05996401]
ouput values: [0.09803108053864205]

Testing Information
------------------------
input values: [0.08]
expected values: [0.07991469]
ouput values: [0.10755662803494578]

Testing Information
------------------------
input values: [0.09]
expected values: [0.08987855]
ouput values: [0.11260655936824379]

Testing Information
------------------------
input values: [0.1]
expected values: [0.09983342]
ouput values: [0.11785758387347331]

Testing Information
------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.12331400152905492]

Testing Information
------------------------
input values: [0.12]
expected values: [0.11971221]
ouput values: [0.12897986925665111]

Testing Information
------------------------
input values: [0.14]
expected values: [0.13954311]
ouput values: [0.14095815502360304]

Testing Information
------------------------
input values: [0.15]
expected values: [0.14943813]
ouput values: [0.1472739226508367]

Testing Information
------------------------
input values: [0.16]
expected values: [0.15931821]
ouput values: [0.15381223489956172]

Testing Information
------------------------
input values: [0.17]
expected values: [0.16918235]
ouput values: [0.1605753912688326]

Testing Information
------------------------
input values: [0.18]
expected values: [0.17902957]
ouput values: [0.16756522283387226]

Testing Information
------------------------
input values: [0.19]
expected values: [0.18885889]
ouput values: [0.1747830513291009]

Testing Information
------------------------
input values: [0.2]
expected values: [0.19866933]
ouput values: [0.18222964797768762]

Testing Information
------------------------
input values: [0.21]
expected values: [0.2084599]
ouput values: [0.18990519246343696]

Testing Information
------------------------
input values: [0.22]
expected values: [0.21822962]
ouput values: [0.1978092325097626]

Testing Information
------------------------
input values: [0.23]
expected values: [0.22797752]
ouput values: [0.20594064460261294]

Testing Information
------------------------
input values: [0.24]
expected values: [0.23770263]
ouput values: [0.21429759646723506]

Testing Information
------------------------
input values: [0.25]
expected values: [0.24740396]
ouput values: [0.22287751197975614]

Testing Information
------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.23167703926014513]

Testing Information
------------------------
input values: [0.27]
expected values: [0.26673144]
ouput values: [0.24069202274916457]

Testing Information
------------------------
input values: [0.28]
expected values: [0.27635565]
ouput values: [0.2499174801138106]

Testing Information
------------------------
input values: [0.29]
expected values: [0.28595223]
ouput values: [0.2593475848487122]

Testing Information
------------------------
input values: [0.3]
expected values: [0.29552021]
ouput values: [0.2689756554401957]

Testing Information
------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.27879415193069984]

Testing Information
------------------------
input values: [0.32]
expected values: [0.31456656]
ouput values: [0.2887946806602098]

Testing Information
------------------------
input values: [0.33]
expected values: [0.32404303]
ouput values: [0.29896800786560335]

Testing Information
------------------------
input values: [0.34]
expected values: [0.33348709]
ouput values: [0.30930408268706727]

Testing Information
------------------------
input values: [0.35]
expected values: [0.34289781]
ouput values: [0.31979206996360937]

Testing Information
------------------------
input values: [0.36]
expected values: [0.35227423]
ouput values: [0.3304203929998614]

Testing Information
------------------------
input values: [0.37]
expected values: [0.36161543]
ouput values: [0.3411767862586958]

Testing Information
------------------------
input values: [0.38]
expected values: [0.37092047]
ouput values: [0.35204835768578924]

Testing Information
------------------------
input values: [0.39]
expected values: [0.38018842]
ouput values: [0.3630216601122283]

Testing Information
------------------------
input values: [0.4]
expected values: [0.38941834]
ouput values: [0.3740827709202294]

Testing Information
------------------------
input values: [0.41]
expected values: [0.39860933]
ouput values: [0.38521737890678615]

Testing Information
------------------------
input values: [0.42]
expected values: [0.40776045]
ouput values: [0.3964108770525109]

Testing Information
------------------------
input values: [0.43]
expected values: [0.4168708]
ouput values: [0.40764845970965663]

Testing Information
------------------------
input values: [0.44]
expected values: [0.42593947]
ouput values: [0.41891522257444974]

Testing Information
------------------------
input values: [0.46]
expected values: [0.44394811]
ouput values: [0.4414485898542462]

Testing Information
------------------------
input values: [0.47]
expected values: [0.45288629]
ouput values: [0.45271375564503413]

Testing Information
------------------------
input values: [0.49]
expected values: [0.47062589]
ouput values: [0.4751559481894903]

Testing Information
------------------------
input values: [0.5]
expected values: [0.47942554]
ouput values: [0.4862917157289635]

Testing Information
------------------------
input values: [0.51]
expected values: [0.48817725]
ouput values: [0.49735777543363896]

Testing Information
------------------------
input values: [0.52]
expected values: [0.49688014]
ouput values: [0.5083418042745044]

Testing Information
------------------------
input values: [0.53]
expected values: [0.50553334]
ouput values: [0.5192321334778778]

Testing Information
------------------------
input values: [0.54]
expected values: [0.51413599]
ouput values: [0.5300177928107933]

Testing Information
------------------------
input values: [0.55]
expected values: [0.52268723]
ouput values: [0.540688544796617]

Testing Information
------------------------
input values: [0.56]
expected values: [0.5311862]
ouput values: [0.5512349090113577]

Testing Information
------------------------
input values: [0.57]
expected values: [0.53963205]
ouput values: [0.5616481768263102]

Testing Information
------------------------
input values: [0.58]
expected values: [0.54802394]
ouput values: [0.5719204171463385]

Testing Information
------------------------
input values: [0.6]
expected values: [0.56464247]
ouput values: [0.5922005117332597]

Testing Information
------------------------
input values: [0.61]
expected values: [0.57286746]
ouput values: [0.6020082510616341]

Testing Information
------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.6116507082823388]

Testing Information
------------------------
input values: [0.63]
expected values: [0.58914476]
ouput values: [0.6211236772824872]

Testing Information
------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.630423619501418]

Testing Information
------------------------
input values: [0.65]
expected values: [0.60518641]
ouput values: [0.6395476299525349]

Testing Information
------------------------
input values: [0.66]
expected values: [0.61311685]
ouput values: [0.6484934011309745]

Testing Information
------------------------
input values: [0.67]
expected values: [0.62098599]
ouput values: [0.6572591855585947]

Testing Information
------------------------
input values: [0.68]
expected values: [0.62879302]
ouput values: [0.6658437576389132]

Testing Information
------------------------
input values: [0.69]
expected values: [0.63653718]
ouput values: [0.6742463754112471]

Testing Information
------------------------
input values: [0.7]
expected values: [0.64421769]
ouput values: [0.6824667427087261]

Testing Information
------------------------
input values: [0.72]
expected values: [0.65938467]
ouput values: [0.6986095941374165]

Testing Information
------------------------
input values: [0.73]
expected values: [0.66686964]
ouput values: [0.7062814211114797]

Testing Information
------------------------
input values: [0.74]
expected values: [0.67428791]
ouput values: [0.7137735052149471]

Testing Information
------------------------
input values: [0.75]
expected values: [0.68163876]
ouput values: [0.7210872611926828]

Testing Information
------------------------
input values: [0.76]
expected values: [0.68892145]
ouput values: [0.7282243486541125]

Testing Information
------------------------
input values: [0.77]
expected values: [0.69613524]
ouput values: [0.7351866448950196]

Testing Information
------------------------
input values: [0.78]
expected values: [0.70327942]
ouput values: [0.7419762196405489]

Testing Information
------------------------
input values: [0.79]
expected values: [0.71035327]
ouput values: [0.7485953116830354]

Testing Information
------------------------
input values: [0.8]
expected values: [0.71735609]
ouput values: [0.7550463073635129]

Testing Information
------------------------
input values: [0.81]
expected values: [0.72428717]
ouput values: [0.7613317208266024]

Testing Information
------------------------
input values: [0.82]
expected values: [0.73114583]
ouput values: [0.7674541759642998]

Testing Information
------------------------
input values: [0.83]
expected values: [0.73793137]
ouput values: [0.7734163899543236]

Testing Information
------------------------
input values: [0.84]
expected values: [0.74464312]
ouput values: [0.7792211582925128]

Testing Information
------------------------
input values: [0.85]
expected values: [0.75128041]
ouput values: [0.7848713412156908]

Testing Information
------------------------
input values: [0.86]
expected values: [0.75784256]
ouput values: [0.7903698514108797]

Testing Information
------------------------
input values: [0.87]
expected values: [0.76432894]
ouput values: [0.7957196429082586]

Testing Information
------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.8009237010583663]

Testing Information
------------------------
input values: [0.89]
expected values: [0.77707175]
ouput values: [0.805985033498366]

Testing Information
------------------------
input values: [0.9]
expected values: [0.78332691]
ouput values: [0.8109066620173666]

Testing Information
------------------------
input values: [0.91]
expected values: [0.78950374]
ouput values: [0.815691615236565]

Testing Information
------------------------
input values: [0.92]
expected values: [0.79560162]
ouput values: [0.8203429220260697]

Testing Information
------------------------
input values: [0.93]
expected values: [0.80161994]
ouput values: [0.8248636055865218]

Testing Information
------------------------
input values: [0.94]
expected values: [0.8075581]
ouput values: [0.8292566781298489]

Testing Information
------------------------
input values: [0.95]
expected values: [0.8134155]
ouput values: [0.8335251360995857]

Testing Information
------------------------
input values: [0.96]
expected values: [0.81919157]
ouput values: [0.8376719558770309]

Testing Information
------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8456611254750959]

Testing Information
------------------------
input values: [0.99]
expected values: [0.83602598]
ouput values: [0.8494595812585194]

Testing Information
------------------------
input values: [1.]
expected values: [0.84147098]
ouput values: [0.853148051480145]

Testing Information
------------------------
input values: [1.01]
expected values: [0.84683184]
ouput values: [0.8567293618292599]

Testing Information
------------------------
input values: [1.02]
expected values: [0.85210802]
ouput values: [0.8602062999755247]

Testing Information
------------------------
input values: [1.03]
expected values: [0.85729899]
ouput values: [0.8635816135915455]

Testing Information
------------------------
input values: [1.04]
expected values: [0.86240423]
ouput values: [0.8668580085981918]

Testing Information
------------------------
input values: [1.05]
expected values: [0.86742323]
ouput values: [0.8700381476148114]

Testing Information
------------------------
input values: [1.06]
expected values: [0.87235548]
ouput values: [0.8731246485986078]

Testing Information
------------------------
input values: [1.08]
expected values: [0.88195781]
ouput values: [0.8790248723208585]

Testing Information
------------------------
input values: [1.09]
expected values: [0.88662691]
ouput values: [0.8818457497805684]

Testing Information
------------------------
input values: [1.1]
expected values: [0.89120736]
ouput values: [0.8845829920705611]

Testing Information
------------------------
input values: [1.11]
expected values: [0.89569869]
ouput values: [0.8872389791663918]

Testing Information
------------------------
input values: [1.12]
expected values: [0.90010044]
ouput values: [0.8898160415101863]

Testing Information
------------------------
input values: [1.13]
expected values: [0.90441219]
ouput values: [0.8923164596576513]

Testing Information
------------------------
input values: [1.14]
expected values: [0.9086335]
ouput values: [0.8947424640355106]

Testing Information
------------------------
input values: [1.15]
expected values: [0.91276394]
ouput values: [0.8970962348027942]

Testing Information
------------------------
input values: [1.16]
expected values: [0.91680311]
ouput values: [0.8993799018097586]

Testing Information
------------------------
input values: [1.17]
expected values: [0.9207506]
ouput values: [0.9015955446485104]

Testing Information
------------------------
input values: [1.18]
expected values: [0.92460601]
ouput values: [0.9037451927896438]

Testing Information
------------------------
input values: [1.19]
expected values: [0.92836897]
ouput values: [0.9058308257994017]

Testing Information
------------------------
input values: [1.2]
expected values: [0.93203909]
ouput values: [0.90785437363205]

Testing Information
------------------------
input values: [1.21]
expected values: [0.935616]
ouput values: [0.909817716992309]

Testing Information
------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9117226877628183]

Testing Information
------------------------
input values: [1.23]
expected values: [0.9424888]
ouput values: [0.9135710694917606]

Testing Information
------------------------
input values: [1.24]
expected values: [0.945784]
ouput values: [0.915364597935884]

Testing Information
------------------------
input values: [1.25]
expected values: [0.94898462]
ouput values: [0.9171049616543052]

Testing Information
------------------------
input values: [1.27]
expected values: [0.95510086]
ouput values: [0.9204026172691673]

Testing Information
------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9219937831204663]

Testing Information
------------------------
input values: [1.29]
expected values: [0.96083506]
ouput values: [0.9235380638187384]

Testing Information
------------------------
input values: [1.3]
expected values: [0.96355819]
ouput values: [0.9250369226662185]

Testing Information
------------------------
input values: [1.31]
expected values: [0.96618495]
ouput values: [0.9264917801539038]

Testing Information
------------------------
input values: [1.32]
expected values: [0.9687151]
ouput values: [0.9279040147925331]

Testing Information
------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.929274963962584]

Testing Information
------------------------
input values: [1.34]
expected values: [0.97348454]
ouput values: [0.9306059247800296]

Testing Information
------------------------
input values: [1.35]
expected values: [0.97572336]
ouput values: [0.9318981549747698]

Testing Information
------------------------
input values: [1.36]
expected values: [0.9778646]
ouput values: [0.9331528737788223]

Testing Information
------------------------
input values: [1.37]
expected values: [0.97990806]
ouput values: [0.9343712628215316]

Testing Information
------------------------
input values: [1.38]
expected values: [0.98185353]
ouput values: [0.9355544670292295]

Testing Information
------------------------
input values: [1.39]
expected values: [0.98370081]
ouput values: [0.9367035955269457]

Testing Information
------------------------
input values: [1.4]
expected values: [0.98544973]
ouput values: [0.9378197225399455]

Testing Information
------------------------
input values: [1.41]
expected values: [0.9871001]
ouput values: [0.9389038882930238]

Testing Information
------------------------
input values: [1.45]
expected values: [0.99271299]
ouput values: [0.9428717477856142]

Testing Information
------------------------
input values: [1.46]
expected values: [0.99386836]
ouput values: [0.9438118451692148]

Testing Information
------------------------
input values: [1.47]
expected values: [0.99492435]
ouput values: [0.9447255250052952]

Testing Information
------------------------
input values: [1.49]
expected values: [0.99673775]
ouput values: [0.9464567937009495]

Testing Information
------------------------
input values: [1.51]
expected values: [0.99815247]
ouput values: [0.9480940105055635]

Testing Information
------------------------
input values: [1.52]
expected values: [0.99871014]
ouput values: [0.9488881512657609]

Testing Information
------------------------
input values: [1.53]
expected values: [0.99916795]
ouput values: [0.9496604330441043]

Testing Information
------------------------
input values: [1.54]
expected values: [0.99952583]
ouput values: [0.9504115333057531]

Testing Information
------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9518373290174107]

Testing Information
------------------------
input values: [1.57]
expected values: [0.99999968]
ouput values: [0.9525289827143266]

Final root-mean-square error: 0.3723425600710058


Process finished with exit code 0


"""