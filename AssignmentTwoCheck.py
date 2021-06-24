"""
Assignment #2 by
   Jonathan Fong
   7/5/2020
   Assignment 1: This program setups up NNData to manage and test our neural network data. The program then tests to see
                 if the NNData class is setup properly.
    Assignment 2: This program adds new attributes (_train_indices, test_indices, _train_pool, and _test_pool) and it
                  adds a new method (split_set()) to the NNData class. Then the program tests to see if additions to the
                  NNData class are working properly.

"""
import numpy as np
from enum import Enum
import collections
import random


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
            features (list): data used to categorize an example
            labels (list): category that fits the features
            train factor (float): percentage of data used for training set
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
        The function to load data into NNData object

        Parameters:
            features (list): data used to categorize an example
            labels (list): category that fits the features
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
            new_train_factor (int): percentage of data used for training set

        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)

        # calculates the number of examples and the number of training examples
        number_of_examples = len(self._features)
        number_of_training_examples = int(self._train_factor * number_of_examples)

        # generates a list of random indirect indices for the training examples
        self._train_indices = random.sample(range(number_of_examples), number_of_training_examples)
        self._train_indices = sorted(self._train_indices)

        # generates a list of random indirect indices for the testing examples
        self._test_indices = [number for number in range(number_of_examples) if not (number in self._train_indices)]
        self._test_indices = sorted(self._test_indices)

    @staticmethod
    def percentage_limiter(percentage: float):
        """
        A static method that limits the training float percentage

        Parameter:
            percentage (float): a float that is used to return a certain value

        Returns:
            int: 0
            float: percentage
            int: 1
        """
        if percentage < 0:
            return 0
        elif 0 <= percentage <= 1:
            return percentage
        else:
            return 1


def unit_test():
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


def main():
    """Runs program"""
    unit_test()


if __name__ == '__main__':
    main()

"""
 "/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/AssignmentTwoCheck.py"
[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
No errors were identified by the unit test.
You should still double check that your code meets spec.
You should also check that PyCharm does not identify any PEP-8 issues.

Process finished with exit code 0
"""