"""
Checkpoint for Assignment Four: Doubly Linked List Submit Assignment
   Jonathan Fong
   7/23/2020
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
   Assignment 4 Check In 1: This program setups up two new classes Node and DoublyLinkedList. The program then test to
                            see if the two classes are setup correctly.

"""


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
                self._curr.next = None
                self._tail = self._curr
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


def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass")
    else:
        print("Fail")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    if my_list.get_current_data() != 0:
        print("Fail")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")


if __name__ == "__main__":
    dll_test()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" 
"/Users/jonathanfong/Desktop/C S 3B (Summer)/AssignmentFourCheck1.py"
Pass
Pass
Pass

Process finished with exit code 0
"""
