"""
Checkpoint: Assignment Three Progress - FFNeurode Class by
   Jonathan Fong
   7/16/2020
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

"""
import numpy as np
from enum import Enum
import collections
import random
import math
from abc import ABC, abstractmethod
import copy


class LayerType(Enum):
    """
    This class determines the role of the neurode.
    """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    """
    This is a parent class for the neurodes.

    Attributes:
        reporting nodes (dictionary) : keeps track of which neighboring nodes, upstream or downstream,
                                       have indicated that they have information available
        reference nodes (dictionary) : represent what the reporting nodes, upstream or downstream, value should be when
                                       all of the nodes have reported
        neighbor (dictionary) : contains references to the neighboring nodes upstream and downstream
    """

    class Side(Enum):
        """
        This class identifies the relationship between neurodes.
        """
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        """
        The constructor for MultiLinkNode Class.
        """
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [], MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        """
        This function returns a representation of the node in a nice format.

        Returns:
            representation_of_node (string) : a string that is a representation of the node
        """
        # title for node representation
        title = f'"Node Information"\n' + f'-----------------\n'

        # id of current node
        current_node_title = 'Current Node:\n'
        id_current_node = str(id(self))

        # id of upstream nodes
        upstream_nodes_title = 'Upstream Node(s):\n'
        id_upstream_nodes_list = [str(id(node)) for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        id_upstream_nodes_string = '; ID:'.join(id_upstream_nodes_list)
        id_upstream_nodes_string = f'ID:{id_upstream_nodes_string}'

        # id of downstream nodes
        downstream_nodes_title = 'Downstream Node(s):'
        id_downstream_nodes_list = [str(id(node)) for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        id_downstream_nodes_string = '; ID:'.join(id_downstream_nodes_list)
        id_downstream_nodes_string = f'ID:{id_downstream_nodes_string}'

        # complete representation of node
        representation_of_node = f'{title}' + f'{current_node_title}' \
                                 + f'\tID:{id_current_node}\n' \
                                 + f'{upstream_nodes_title}' \
                                 + f'\t{id_upstream_nodes_string}\n' \
                                 + f'{downstream_nodes_title}\n' \
                                 + f'\t{id_downstream_nodes_string}\n'
        return representation_of_node

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """
        This function is an abstract method that must be implemented by subclasses of MultiLinkNeurode.

        Parameters:
            node (MultiLinkNode) : basic unit of data structure
            side (Side) : relationship between neurodes (upstream or downstream)
        """
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """
        This function resets (or sets) the nodes that link into this node either upstream or downstream.

        Parameters:
            nodes (list) : list of basic units of data structure
            side (Side) : relationship between neurodes (upstream or downstream)
        """
        # copies the nodes parameter into the appropriate entry of self._neighbors
        if side == MultiLinkNode.Side.UPSTREAM:
            self._neighbors[MultiLinkNode.Side.UPSTREAM] = copy.copy(nodes)
        elif side == MultiLinkNode.Side.DOWNSTREAM:
            self._neighbors[MultiLinkNode.Side.DOWNSTREAM] = copy.copy(nodes)
        else:
            raise TypeError('nodes must be a list and side must be a Side '
                            '(MultiLinkNode.Side.UPSTREAM or MultiLinkNode.Side.DOWNSTREAM)')

        # calls _process_new_neighbor() for each node
        for node in nodes:
            self._process_new_neighbor(node, side)

        # calculates and store the appropriate reference value in the correct element of self._reference_value
        nodes_list_length = len(nodes)
        if side == MultiLinkNode.Side.UPSTREAM:
            self._reference_value[MultiLinkNode.Side.UPSTREAM] = (1 << nodes_list_length) - 1
        elif side == MultiLinkNode.Side.DOWNSTREAM:
            self._reference_value[MultiLinkNode.Side.DOWNSTREAM] = (1 << nodes_list_length) - 1
        else:
            raise TypeError('nodes must be a list and side must be a Side '
                            '(MultiLinkNode.Side.UPSTREAM or MultiLinkNode.Side.DOWNSTREAM)')


class Neurode(MultiLinkNode):
    """
    This is a class that inherits from MultiLinkNode and sets up neurodes.

    Attributes:
        value (integer) : current value of the neurode
        node type (LayerTyoe) : represents the role of nuerode
        learning rate (float) : learning rate used in backpropogation
        weights (dictionary) : holds references to upstream neurodes and those neurode's weights
    """

    def __init__(self, node_type, learning_rate=0.05):
        """
        The constructor for Neurode class.

        Parameters:
            node type (LayerType) : represents the role of nuerode
            learning rate (float) : learning rate used in backpropogation
        """
        self._value = 0
        self._node_type = node_type
        self._learning_rate = Neurode.learning_rate_limiter(learning_rate)
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        """
        This function is called when new neighbors are added and adds the node and its weight to neighbors.

        Parameters:
            node (MultiLinkNode) : basic unit of data structure
            side (Side) : relationship between neurodes (upstream or downstream)
        """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side):
        """
        This function indicates that a neighboring node has information available and has reported.

        Parameters:
            node (MultiLinkNode) : basic unit of data structure
            side (Side) : relationship between neurodes (upstream or downstream)

        Returns:
            True : boolean representing that all the neighboring nodes have reported
            False: boolean representing that not all the neighboring nodes have reported
        """
        if side == MultiLinkNode.Side.UPSTREAM:
            node_index = self._neighbors[MultiLinkNode.Side.UPSTREAM].index(node)
            self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] = \
                self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] | (1 << node_index)
            if self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == self._reference_value[MultiLinkNode.Side.UPSTREAM]:
                self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] = 0
                return True
            return False
        elif side == MultiLinkNode.Side.DOWNSTREAM:
            node_index = self._neighbors[MultiLinkNode.Side.DOWNSTREAM].index(node)
            self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = \
                self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] | (1 << node_index)
            if self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] == \
                    self._reference_value[MultiLinkNode.Side.DOWNSTREAM]:
                self._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 0
                return True
            return False
        else:
            raise TypeError('node must be a Neurode and side must be a Side '
                            '(MultiLinkNode.Side.UPSTREAM or MultiLinkNode.Side.DOWNSTREAM)')

    def get_weight(self, node):
        """
        This function gets the upstream node's associated weight.

        Parameters:
            node (MultiLinkNode) : basic unit of data structure

        Returns:
            upstream_node_weight (float) : associated upstream node's weight
        """
        upstream_node_weight = self._weights[node]
        return upstream_node_weight

    @property
    def value(self):
        """
        A property that returns the node's value.

        Returns:
            value (integer) : current value of the neurode
        """
        return self._value

    @property
    def node_type(self):
        """
        A property that returns the node's type.

        Returns:
            node type (LayerType) : represents the role of nuerode
        """
        return self._node_type

    @property
    def learning_rate(self):
        """
        A property that returns the node's learning rate.

        Returns:
            learning rate (float) : learning rate used in backpropogation
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """
        A setter that sets the learning rate.

        Parameters:
            learning rate (float) : learning rate used in backpropogation
        """
        self._learning_rate = Neurode.learning_rate_limiter(learning_rate)

    @staticmethod
    def learning_rate_limiter(learning_rate: float):
        """
        A static method that limits the learning rate.

        Parameters:
            learning rate (float) : learning rate used in backpropogation

        Returns:
            0 : integer representing learning rate
            learning rate (float) : float representing learning rate
            1: integer representing learning rate
        """
        if learning_rate < 0:
            return 0
        elif 0 <= learning_rate <= 1:
            return learning_rate
        else:
            return 1


class FFNeurode(Neurode):
    def __init__(self, my_type):
        """
        The constructor for FFNeurode Class

        Parameters:
             my_type (LayerType) : represents the role of nuerode
        """
        super().__init__(my_type)

    def set_input(self, input_value):
        """
        The function that set the value of an input layer neurode.

        Parameters:
            input value (float) : value of an input layer neurode
        """
        self._value = input_value
        for downstream_node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream_node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """
        The function that upstream neurodes call when they have data ready.

        Parameters:
            neurode (FFNeurode) : upstream layer neurode
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def _calculate_value(self):
        """
        The function that calculates the upstream node's values.
        """
        # creates a list of the upstream node's values and upstream node's weights
        upstream_neurodes_value = [upstream_node._value for upstream_node in
                                   self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        upstream_neurodes_weight = list(self._weights.values())

        # calculates the weighted sum of the upstream node's values and weights
        for index in range(len(upstream_neurodes_weight)):
            self._value = self._value + (upstream_neurodes_weight[index] * upstream_neurodes_value[index])

        # the result of the sigmoid function with the weighted sum passed to it
        self._value = self._sigmoid(self._value)

    def _fire_downstream(self):
        """
        The function that calls data_ready_upstream on the current node's downstream neighbors.
        """
        for downstream_node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            downstream_node.data_ready_upstream(self)

    @staticmethod
    def _sigmoid(value):
        """
        The static method that finds the sigmoid of a given value.

        Parameters:
            value (float) : a value that will be converted by the sigmoid function

        Returns:
            sigmoid answer (float) : result of passing the value into the sigmoid function
        """
        sigmoid_answer = 1 / (1 + np.exp(-value))
        return sigmoid_answer


def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")


def main():
    """Runs program"""
    check_point_two_test()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/Test.py"
0.6471688100770087 0.6471688100770087

Process finished with exit code 0
"""
