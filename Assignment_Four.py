"""
Assignment Four: LayerList
   Jonathan Fong
   7/26/2020
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
                  is setup correctly

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
    """
    This is a class that inherits from Neurode and setups and setups feed forward neurodes.
    """

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
            node (Neurode) : upstream layer neurode
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def _calculate_value(self):
        """
        The function that calculates the upstream node's values.
        """
        self._value = 0
        for upstream_neurode in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            self._value += self.get_weight(upstream_neurode) * upstream_neurode.value
        self._value = FFNeurode._sigmoid(self._value)

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


class BPNeurode(Neurode):
    """
    The class that inherits from Neurode and setups back propogation neurodes
    """

    def __init__(self, my_type):
        """
        The constructor for the BPNeurode class

        Attributes:
            delta (float) : change in output value in terms of weight
        """
        self._delta = 0
        super().__init__(my_type)

    def set_expected(self, expected_value):
        """
        The function that sets the delta value for output layer nodes.

        Parameters:
            expected value (float) : the value that an output node should return
        """
        self._calculate_delta(expected_value)
        for upstream_node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            upstream_node.data_ready_downstream(self)

    def data_ready_downstream(self, node):
        """
        The function that downstream neurodes call when they have data ready.

        Parameters:
            node (Neurode) : downstream layer neurode
        """
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def _calculate_delta(self, expected_value=None):
        """
        The function that calculates the delta of a neurode.

        Parameters:
            expected value (float) : the value that an output node should return
        """
        self._delta = 0
        if (self.node_type == LayerType.INPUT) or (self._node_type == LayerType.HIDDEN):
            for downstream_node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += downstream_node.get_weight(self) * downstream_node.delta
            self._delta *= BPNeurode._sigmoid_derivative(self._value)
        elif self.node_type == LayerType.OUTPUT:
            self._delta = (expected_value - self._value) * BPNeurode._sigmoid_derivative(self._value)
        else:
            raise TypeError('neurode must be a an input, hidden, or output (Layer.INPUT, Layer.HIDDEN or Layer.OUTPUT) '
                            'node')

    def _fire_upstream(self):
        """"
        The function that calls data_ready_upstream on the current node's downstream neighbors.
        """
        for upstream_node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            upstream_node.data_ready_downstream(self)

    def _update_weights(self):
        """
        The function that calculates the new weights.
        """
        for downstream_node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = downstream_node.get_weight(self) + (
                    self.value * downstream_node.delta * downstream_node.learning_rate)
            downstream_node.adjust_weights(self, adjustment)

    def adjust_weights(self, node, adjustment):
        """
        The function that adjust the old weights to the new weights.

        Parameters:
            node (Neurode) : upstream layer neurode
            adjustment (float) : new weight
        """
        self._weights[node] = adjustment

    @staticmethod
    def _sigmoid_derivative(value):
        """
        The static method that calculates the sigmoid derivative of the neurode value.

        Parameters:
            value (float) : a value that will be converted by the sigmoid derivative function
        """
        sigmoid_derivative_answer = value * (1 - value)
        return sigmoid_derivative_answer

    @property
    def delta(self):
        """
        The property that returns a neurode's delta value.

        Returns:
            delta (float) : change in output value in terms of weight
        """
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    """
    The class that inherits from FFNeurode and BPNeurode and setups up
    """
    pass


class Node:
    """
    This class setups nodes.

    Attributes:
        data : any type of data
        next (Node) : next node
        previous (Node) previous node
    """
    def __init__(self, data=None):
        """
        The constructor for Neurode:

        Parameters:
            data : any type of data
        """
        self.data = data
        self.next = None
        self.previous = None


class DoublyLinkedList:
    """
    This class setups doubly linked list.

    Attributes:
        head (Node) : top node
        tail (Node) : bottom node
        curr (Node) : current node
        curr iter (Node) : current node used for the iterator
    """
    def __init__(self):
        """
        The constructor for DoublyLinkedList
        """
        self._head = None
        self._tail = None
        self._curr = None
        self._curr_iter = None

    class EmptyListError(Exception):
        """
        This class is a user defined exception class called EmptyListError and is called when a doubly linked list is
        empty.
        """
        pass

    def add_to_head(self, data):
        """
        This function creates a new head for the doubly linked list'

        Parameters:
            data : ant type of data
        """
        new_node = Node(data)

        new_node.next = self._head
        if new_node.next is not None:
            new_node.next.previous = new_node

        if self._head is None:
            self._tail = new_node
        self._tail.next = None
        self._head = new_node

        self.reset_to_head()

    def remove_from_head(self):
        """
        This function removes a head from the doubly linked list.

        Returns:
            remove_head_data : the data of the head that is being removed
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError('Cannot remove the head because the doubly linked list is empty')
        elif self._head == self._tail:
            remove_head_data = self._head.data

            self._head = None
            self._tail = None
            self._curr = None

            return remove_head_data
        else:
            remove_head_data = self._head.data

            if (self._head.next != self._tail) or (self._head.next is not None):
                self._head.next.previous = self._head.previous
            self._head = self._head.next

            self.reset_to_head()

            return remove_head_data

    def move_forward(self):
        """
        This function moves the position forward in the doubly linked list.

        Returns:
            current data : current node data
        """
        # moves position forward in the doubly linked list
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot move forward because the doubly linked list is empty')
        elif (self._curr == self._tail) or (self._curr.next is None):
            raise IndexError('Cannot go past the end of the doubly linked list')
        else:
            self._curr = self._curr.next

        # returns the current node's data
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('The doubly linked list is empty')
        else:
            current_data = self._curr.data
            return current_data

    def move_back(self):
        """"
        This function moves the position backward in the doubly linked list.

        Returns:
            current data : current node data
        """
        # moves position backward in the doubly linked list
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot move backward because the doubly linked list is empty')
        elif (self._curr == self._head) or (self._curr.previous is None):
            raise IndexError('Cannot go past the beginning of the doubly linked list')
        else:
            self._curr = self._curr.previous

        # returns the current node's data
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('The doubly linked list is empty')
        else:
            current_data = self._curr.data
            return current_data

    def add_after_cur(self, data):
        """
        This function adds nodes to the doubly linked list.

        Parameters:
            data : any type of data
        """
        if self._curr is None:
            self.add_to_head(data)
            return
        elif (self._curr == self._tail) or (self._curr.next is None):
            new_node = Node(data)

            new_node.previous = self._curr
            self._curr.next = new_node

            self._tail = new_node
        else:
            new_node = Node(data)

            self._curr.next.previous = new_node
            new_node.next = self._curr.next
            self._curr.next = new_node
            new_node.previous = self._curr

    def remove_after_cur(self):
        """
        This function removes nodes to the doubly linked list.

        Returns:
            remove after curr data : the data of the node that is being removed
        """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot remove; the doubly linked list is empty')
        elif self._head == self._tail:
            remove_curr_data = self._curr.data

            self.remove_from_head()

            return remove_curr_data
        elif (self._curr == self._tail) or (self._curr.next is None):
            raise IndexError('You have reached the end of the doubly linked list')
        else:
            if (self._curr.next == self._tail) or (self._curr.next is None):
                remove_after_curr_data = self._curr.next.data

                self._curr.next = None
                self._tail = self._curr

                return remove_after_curr_data
            else:
                remove_after_curr_data = self._curr.next.data

                self._curr.next.next.previous = self._curr
                self._curr.next = self._curr.next.next

                return remove_after_curr_data

    def reset_to_head(self):
        """
        This function sets the current position to the head of doubly linked list.

        Returns:
            current data head : current node (head node) data
        """
        # sets current position to the head of doubly linked list
        self._curr = self._head

        # returns the current node's (head node) data
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot reset to head because the doubly linked list is empty')
        else:
            current_data_head = self._curr.data
            return current_data_head

    def reset_to_tail(self):
        """
        This function sets the current position to the tail of doubly linked list.

        Returns:
            current data tail : current node (tail node) data
        """
        # sets current position to the tail of doubly linked list
        self._curr = self._tail

        # returns the current node's (tail node) data
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot reset to tail because the doubly linked list is empty')
        else:
            current_data_tail = self._curr.data
            return current_data_tail

    def get_current_data(self):
        """
        This function returns the current node's data.

        Returns:
            current data : current node's data
        """
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError('Cannot return data because the doubly linked list is empty')
        else:
            current_data = self._curr.data
            return current_data

    def __iter__(self):
        """
        The function that creates an iterable object.

        Returns:
            self (DoublyLinkedList) : iterable object
        """
        self._curr_iter = self._head
        return self

    def __next__(self):
        """
        The function that iterates through the DoublyLinkedList iterable object.

        Return:
            curr iter data : node's data at current iter position
        """
        if self._curr_iter is None:
            raise StopIteration
        curr_iter_data = self._curr_iter.data
        self._curr_iter = self._curr_iter.next
        return curr_iter_data


class LayerList(DoublyLinkedList):
    """
    This class inherits from DoublyLinkedList and setups a doubly linked list for neurodes and their layers.

    Attributes:
        input nodes (FFBPNeurode) : input neurodes
        output nodes (FFBPNeurode) : output neurodes
    """
    def __init__(self, inputs: int, outputs: int):
        """
        The constructor for LayerList.

        Parameters:
            inputs (integer) : number of input nodes
            outputs (integer) : number of output nodes
        """
        if (type(inputs) == int) and (type(outputs) == int):
            super().__init__()

            # setups _input_node and _output_node attributes
            self._input_nodes = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
            self._output_nodes = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]

            # links the input and output layers together
            for input_node in self.input_nodes:
                input_node.reset_neighbors(self.output_nodes, MultiLinkNode.Side.DOWNSTREAM)
            for output_node in self.output_nodes:
                output_node.reset_neighbors(self.input_nodes, MultiLinkNode.Side.UPSTREAM)

            # adds the input and output layers to the LayerList
            self.add_to_head(self.input_nodes)
            self.add_after_cur(self.output_nodes)
        else:
            raise TypeError('You must pass an integer for number of input and output neurodes')

    def add_layer(self, num_nodes: int):
        """
        The function that adds hidden layers to the LayerList.

        Parameters:
            num_nodes (int) : number of hidden nodes
        """
        if type(num_nodes) == int:
            if (self._curr == self._tail) or (self._curr.next is None):
                raise IndexError('Cannot add layer after output layer')
            else:
                # setups hidden nodes
                hidden_nodes = [FFBPNeurode(LayerType.HIDDEN) for _ in range(num_nodes)]

                # adds hidden layer to LayerList
                self.add_after_cur(hidden_nodes)

                # links the hidden layer with upstream and downstream neighbors
                for upstream_node in self._curr.data:
                    upstream_node.reset_neighbors(hidden_nodes, MultiLinkNode.Side.DOWNSTREAM)
                for hidden_node in hidden_nodes:
                    hidden_node.reset_neighbors(self._curr.data, MultiLinkNode.Side.UPSTREAM)
                    hidden_node.reset_neighbors(self._curr.next.next.data, MultiLinkNode.Side.DOWNSTREAM)
                for downstream_node in self._curr.next.next.data:
                    downstream_node.reset_neighbors(hidden_nodes, MultiLinkNode.Side.UPSTREAM)
        else:
            raise TypeError('You must pass an integer for the number of hidden neurodes')

    def remove_layer(self):
        """
        The function that removes hidden layers from the LayerList
        """
        if (self._curr.next == self._tail) or (self._curr == self._tail.previous):
            raise IndexError('Cannot remove output layer')
        else:
            # removes hidden layer from LayerList
            self.remove_after_cur()

            # links the remaining upstream and downstream layers
            for upstream_node in self._curr.data:
                upstream_node.reset_neighbors(self._curr.next.data, MultiLinkNode.Side.DOWNSTREAM)
            for downstream_node in self._curr.next.data:
                downstream_node.reset_neighbors(self._curr.data, MultiLinkNode.Side.UPSTREAM)

    @property
    def input_nodes(self):
        """
        A property that returns the list of input nodes.

        Returns:
            input nodes (list) : list of input nodes
        """
        return self._input_nodes

    @property
    def output_nodes(self):
        """
        A property that returns the list of output nodes.

        Returns:
            output nodes (list) : list of output nodes
        """
        return self._output_nodes

# def new_test():
#     my_list = LayerList(2, 4)
#     my_list.add_layer(3)
#     my_list.add_layer(6)
#     my_list.reset_to_head()
#     print(my_list._curr.data[0].node_type)
#     my_list.move_forward()
#     print(my_list.get_current_data()[0].node_type, LayerType.HIDDEN)
#     my_list.move_forward()
#     print(my_list._curr.data[0].node_type)
#     my_list.move_forward()
#     print(my_list._curr.data[0].node_type)

def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    print(my_list.get_current_data()[0].node_type, LayerType.HIDDEN)
    print(LayerType.HIDDEN.value)
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        my_list.remove_layer()
        assert False
    except IndexError:
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value


if __name__ == "__main__":
    layer_list_test()
    # new_test()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" 
"/Users/jonathanfong/Desktop/C S 3B (Summer)/Assignment_Four.py"

Process finished with exit code 0
"""
