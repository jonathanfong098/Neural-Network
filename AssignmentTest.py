import numpy as np
from enum import Enum
import collections
import random
import math
from abc import ABC, abstractmethod
import copy


class DataMismatchError(Exception):
    """This class is a user defined exception called DataMismatchError"""
    pass


class NNData:
    """
    This is a class for manging and testing our neural network data.

    Attributes:
        train factor (float): percentage of data used in training set
        train indices (list): indirect indices for training examples
        test indices (list): indirect indices for testing examples
        train pool (deque): keeps track of training items that have not been seen in a particular training period
        test pool (deque): keeps track of which training items have not been seen in a testing run
        features (list): data used to categorize an example
        labels (list): category that fits the features
    """

    class Order(Enum):
        """
        This class determines whether the training data is presented in the same order or random order.
        """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """
        This class identifies whether the user is requesting training set or testing set data.
        """
        TRAIN = 0
        TEST = 1

    def __init__(self, features=None, labels=None, train_factor=0.9):
        """
        The constructor for NNData Class.

        Parameters:
            features (list) : data used to categorize an example
            labels (list) : category that fits the features
            train factor (float) : percentage of data used for training set
        """
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque()
        self._test_pool = collections.deque()

        if features is None:
            features = []
        self._features = None

        if labels is None:
            labels = []
        self._labels = None

        try:
            self.load_data(features, labels)
            if features is not None and labels is not None:
                self.split_set()
        except (ValueError, DataMismatchError):
            pass

    def load_data(self, features=None, labels=None):
        """
        The function to load data into NNData object.

        Parameters:
            features (list) : data used to categorize an example
            labels (list) : category that fits the features
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return

        if len(features) != len(labels):
            raise DataMismatchError('Features and labels lists are different lengths')

        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError('Label and feature lists must be homogeneous (same data type)'
                             'and numeric (i.e integers and floats) list of lists')

    def split_set(self, new_train_factor=None):
        """
        The function that sets up the indirect indices lists (_train_indices and _test indices)
        depending on the user set training factor

        Parameters:
            new_train_factor (int) : percentage of data used for training set
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)

        # calculates the number of examples and the number of training examples
        number_of_examples = len(self._features)
        number_of_training_examples = math.floor((self._train_factor * number_of_examples))

        # generates a list of random indirect indices for the training examples
        self._train_indices = random.sample(range(number_of_examples), number_of_training_examples)
        self._train_indices.sort()

        # generates a list of random indirect indices for the testing examples
        self._test_indices = [number for number in range(number_of_examples) if not (number in self._train_indices)]
        self._test_indices.sort()

    def prime_data(self, target_set=None, order=None):
        """
        The function will load one or both pools (_train_pool and _test_pool) to be used as indirect indices.

        Parameters:
            target_set (enum object) : determines whether user loads _train_pool, _test_pool, or both
        """
        # clear both pools (_train_pool and _test_pool) to prevent over-filling
        self._train_pool.clear()
        self._test_pool.clear()

        # load one or both pools (_train_pool and _test_pool)
        if target_set is None:
            self._train_pool.extend(self._train_indices)
            self._test_pool.extend(self._test_indices)
        elif target_set == NNData.Set.TRAIN:
            self._train_pool.extend(self._train_indices)
        elif target_set == NNData.Set.TEST:
            self._test_pool.extend(self._test_indices)
        else:
            raise IndexError('target_set must be NNData.Set.TRAIN, NNData.Set.TEST, or None')

        # shuffles indices in the pools or keeps indices, in the pools, in sequential order
        if order == NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        elif (order == NNData.Order.SEQUENTIAL) or (order is None):
            pass
        else:
            raise IndexError('target_set must be NNData.Set.TRAIN, NNData.Set.TEST, or None')

    def get_one_item(self, target_set=None):
        """
        This function will return one feature and label pair as a tuple.

        Parameter:
            target_set (enum) : determines whether _train_pool or _test_pool is used to create the
                                feature and label tuple

        Returns:
            item : a tuple that has a feature and label pair

        """
        if (target_set == NNData.Set.TRAIN) or (target_set is None):
            try:
                index = self._train_pool.popleft()
                item = (self._features[index], self._labels[index])
                return item
            except IndexError:
                return None
        elif target_set == NNData.Set.TEST:
            try:
                index = self._test_pool.popleft()
                item = (self._features[index], self._labels[index])
                return item
            except IndexError:
                return None
        else:
            raise IndexError('target_set must be NNData.Set.TRAIN, NNData.Set.TEST, or None')

    def number_of_samples(self, target_set=None):
        """
        A function that returns the total number of training examples, the total number of testing examples, or the
        total number of examples.

        Parameters:
            target_set (enum object) : determines whether the total number of training examples, total number of
                                       testing examples, or total number of examples is returned
        Return:
            number_of_training_examples : total number of training examples
            number_of_testing_examples : total number of testing examples
            total_number_of_examples : total number of examples
        """
        number_of_training_examples = len(self._train_indices)
        number_of_testing_examples = len(self._test_indices)
        total_number_of_examples = len(self._features)

        if target_set == NNData.Set.TRAIN:
            return number_of_training_examples
        elif target_set == NNData.Set.TEST:
            return number_of_testing_examples
        elif target_set is None:
            return total_number_of_examples
        else:
            raise IndexError('target_set must be NNData.Set.TRAIN, NNData.Set.TEST, or None')

    def pool_is_empty(self, target_set=None):
        """
        The function that tells whether the pools (_train_pool or _test_pool) are empty.

        Parameter:
            target_set (enum object) : determines which pool to check for emptiness

        Return:
            True: indicates that _train_pool or test_pool is empty
            False: indicates that _train_pool or test_pool is not empty
        """
        if (target_set == NNData.Set.TRAIN) or (target_set is None):
            if len(self._train_pool) == 0:
                return True
            else:
                return False
        elif target_set == NNData.Set.TEST:
            if len(self._test_pool) == 0:
                return True
            else:
                return False
        else:
            raise IndexError('target_set must be NNData.Set.TRAIN, NNData.Set.TEST, or None')

    @staticmethod
    def percentage_limiter(percentage: float):
        """
        A static method that limits the training float percentage

        Parameter:
            percentage (float): a float that is used to return a certain value

        Returns:
            0 : integer representing 0%
            percentage : float representing a percentage between 0 and 100
            1: integer representing 100%
        """
        if percentage < 0:
            return 0
        elif 0 <= percentage <= 1:
            return percentage
        else:
            return 1


def load_XOR():
    """loads XOR data into NNData object"""
    xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_labels = [[0], [1], [1], [0]]
    xor = NNData(xor_features, xor_labels, 1)
    return xor


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
        title = f'"Node Information"\n' + f'-----------------\n'

        current_node_title = 'Current Node:\n'
        id_current_node = str(id(self))

        upstream_nodes_title = 'Upstream Node(s):\n'
        id_upstream_nodes_list = [str(id(node)) for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        id_upstream_nodes_string = '; ID:'.join(id_upstream_nodes_list)
        id_upstream_nodes_string = f'ID:{id_upstream_nodes_string}'

        downstream_nodes_title = 'Downstream Node(s):'
        id_downstream_nodes_list = [str(id(node)) for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        id_downstream_nodes_string = '; ID:'.join(id_downstream_nodes_list)
        id_downstream_nodes_string = f'ID:{id_downstream_nodes_string}'

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
        This function is an abstract method that must be implemented by subclasses of MultiLinkNeurode

        Parameters:
            node (MultiLinkNode) : basic unit of data structure
            side (Side) : relationship between neurodes (upstream or downstream)
        """
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        if side == MultiLinkNode.Side.UPSTREAM:
            self._neighbors[MultiLinkNode.Side.UPSTREAM] = copy.copy(nodes)
        elif side == MultiLinkNode.Side.DOWNSTREAM:
            self._neighbors[MultiLinkNode.Side.DOWNSTREAM] = copy.copy(nodes)
        else:
            raise TypeError('nodes must be a list and side must be a Side '
                            '(MultiLinkNode.Side.UPSTREAM or MultiLinkNode.Side.DOWNSTREAM)')

        for node in nodes:
            self._process_new_neighbor(node, side)

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
        The constructor for Neurode class

        Parameters:
            node type (LayerTyoe) : represents the role of nuerode
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
        upstream_neurodes_value = [upstream_node._value for upstream_node in
                                   self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        upstream_neurodes_weight = list(self._weights.values())

        for index in range(len(upstream_neurodes_weight)):
            self._value = self._value + (upstream_neurodes_weight[index] * upstream_neurodes_value[index])

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
        sigmoid_answer = 1 / (1 + math.exp(-value))
        return sigmoid_answer


def unit_test_assignment1():
    """Tests if NNData class is working properly"""

    # Tests for DataMismatchError and verifies if self._feature == None and self._labels == None
    data_mismatch_error_test = load_XOR()

    features_different_length = [0, 1]
    labels_different_length = [0]

    try:
        data_mismatch_error_test.load_data(features_different_length, labels_different_length)
    except DataMismatchError:
        data_mismatch_error_test = NNData(features_different_length, labels_different_length)
        print(f'If features and labels have different lengths then a DataMismatchError is raised and '
              f'self._features and self._labels are {data_mismatch_error_test._features}, '
              f'{data_mismatch_error_test._labels}\n'
              )

    # Tests for Value Error and verifies if self._feature == None and self._labels == None
    value_error_test = load_XOR()
    features_bad_data = ['string1', 'string2']
    labels_bad_data = ['string3', 'string4']

    try:
        value_error_test.load_data(features_bad_data, labels_bad_data)
    except ValueError:
        value_error_test = NNData(features_bad_data, labels_bad_data)
        print(f'If features and labels contain non float values then a ValueError is raised'
              f'and self._features and self._labels are {value_error_test._features}, {value_error_test._labels}\n'
              )

    # Tests if invalid data is passed to NNData that self._feature == None and self._labels == None
    constructor_invalid_data_1 = NNData([0], [0, 1])
    print(f'If invalid data is passed to the constructors (i.e a list with different lengths) then '
          f'self._features and self._labels are {constructor_invalid_data_1._features}, '
          f'{constructor_invalid_data_1._labels}\n'
          )

    constructor_invalid_data_2 = NNData('test string1', 'test string2')
    print(f'If invalid data is passed to the constructors (i.e a list that cannot be made into a homogenous array '
          f'of floats) then self._features and self._labels are {constructor_invalid_data_2._features}, '
          f'{constructor_invalid_data_2._labels}\n'
          )

    # Tests that NNData limits the training factor to zero if a negative number is passed
    train_factor_zero = NNData([0, 1], [0, 1], -1)
    print(f'If a negative number is passed into NNData then train factor is {train_factor_zero._train_factor}\n')

    # Tests that NNData limits the training factor to one if a number greater than one is passed
    train_factor_one = NNData([0, 1], [0, 1], 2)
    print(f'If a positive number is passed into NNData then train factor is {train_factor_one._train_factor}')


def unit_test_assignment2_checkpoint():
    """Tests if NNData class is working properly"""
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        print(our_data_0._features)
        x = list(range(100))
        y = x
        our_big_data = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of samples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True  # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


def unit_test_assignment2():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = list(range(10))
        y = x
        our_data_0 = NNData(x, y)
        print(our_data_0._features)
        x = list(range(100))
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [1]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [1, 2, 3, 4]
        y = [.1, .2, .3, .4]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(my_x_list) == set(x)
        assert set(my_y_list) == set(y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


def unit_test_assignment3_checkpoint1():
    # Mock up a network with three inputs and three outputs

    inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
    outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
        print("Fail - Initial reference value is not zero")
    for node in inputs:
        node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
    for node in outputs:
        node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
        print("Fail - Final reference value is not correct")
    if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Final reference value is not correct")

    # Report data ready from each input and make sure _check_in
    # only reports True when all nodes have reported

    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Initial reporting value is not zero")
    if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Report data ready from each output and make sure _check_in
    # only reports True when all nodes have reported

    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Check that learning rates were set correctly

    if not inputs[0].learning_rate == .05:
        print("Fail - default learning rate was not set")
    if not outputs[0].learning_rate == .01:
        print("Fail - specified learning rate was not set")

    # Check that weights appear random

    weight_list = list()
    for node in outputs:
        for t_node in inputs:
            if node.get_weight(t_node) in weight_list:
                print("Fail - weights do not appear to be set up properly")
            weight_list.append(node.get_weight(t_node))

    print('assignment 3 checkpoint 1 works')


def unit_test_assignment3_checkpoint2():
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

    print('assignment 3 checkponint 2 works')



def main():
    """Runs program"""
    unit_test_assignment1()
    print()
    unit_test_assignment2_checkpoint()
    print()
    unit_test_assignment2()
    print()
    unit_test_assignment3_checkpoint1()
    print()
    unit_test_assignment3_checkpoint2()


if __name__ == '__main__':
    main()
