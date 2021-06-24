"""
Assignment #2 by
   Jonathan Fong
   7/9/20202
   Assignment 1: This program setups up NNData to manage and test our neural network data. The program then tests to see
                 if the NNData class is setup properly.
   Assignment 2 Check: This program adds new attributes (_train_indices, test_indices, _train_pool, and _test_pool)
                       and it adds a new method (split_set()) to the NNData class. Then the program tests to see if
                       the additions to the NNData class are working properly.
   Assignment 2: This program adds four new methods, prime_data(), get_one_item(), number_of_samples(), and
                  pool_is_empty(), to the NNData class. Then the program tests to see if the the additions to the
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
        number_of_training_examples = int(self._train_factor * number_of_examples)

        # generates a list of random indirect indices for the training examples
        self._train_indices = random.sample(range(number_of_examples), number_of_training_examples)
        self._train_indices = sorted(self._train_indices)

        # generates a list of random indirect indices for the testing examples
        self._test_indices = [number for number in range(number_of_examples) if not (number in self._train_indices)]
        self._test_indices = sorted(self._test_indices)

    def prime_data(self, target_set=None, order=None):
        """
        The function will load one or both pools (_train_pool and _test_pool) to be used as indirect indices.

        Parameters:
            target_set (enum object) : determines whether user loads _train_pool, _test_pool, or both
        """
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
        total_number_of_examples = number_of_training_examples + number_of_testing_examples

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
        A function that limits the training float percentage

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


def unit_test():
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
            print(our_data_1._features, our_data_1._labels)
            example = our_data_1.get_one_item()
            print(our_data_1._features, our_data_1._labels)
            print(example)
            my_x_list.append(example[0])
            my_y_list.append(example[1])
            # print(our_data_1._features, our_data_1._labels)
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


def main():
    """Runs program"""
    unit_test()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/Assignment_Two.py"
[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
No errors were identified by the unit test.
You should still double check that your code meets spec.
You should also check that PyCharm does not identify any PEP-8 issues.

Process finished with exit code 0
"""