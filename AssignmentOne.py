"""
Assignment #1 by
   Jonathan Fong
   7/2/20202
   Assignment 1: This program setups up NNData to manage and test our neural network data. The program then test to see
                 if the NNData class is setup properly.
"""
import numpy as np
from enum import Enum


class DataMismatchError(Exception):
    """This class is a user defined exception called DataMismatchError"""
    pass


class NNData:
    """
    This is a class for manging and testing our neural network data.

    Attributes:
        features (list): data used to categorize an example
        labels (list): category that fits the features
        train factor (float): percentage of data used in training set

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
        TEST = 0
        TRAIN = 1

    def __init__(self, features=None, labels=None, train_factor=0.9):
        """
        The constructor for NNData Class.

        Parameters:
            features (list): data used to categorize an example
            labels (list): category that fits the features
            train factor (float): percentage of data used in training set
        """
        try:
            self._features = None
            self._labels = None
            self._train_factor = NNData.percentage_limiter(train_factor)

            if features is None:
                features = []
            else:
                self._features = None

            if labels is None:
                labels = []
            else:
                self._labels = None

            self.load_data(features, labels)

        except (ValueError, DataMismatchError):
            self._features = None
            self._labels = None

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

    @staticmethod
    def percentage_limiter(percentage: float):
        """
        A function that limits the training float percentage

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


def load_XOR():
    """loads XOR data into NNData object"""
    xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_labels = [[0], [1], [1], [0]]
    xor = NNData(xor_features, xor_labels, 1)
    return xor


def unit_test():
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


def main():
    """Runs program"""
    unit_test()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/AssignmentOne.py"
If features and labels have different lengths then a DataMismatchError is raised and self._features and self._labels are None, None

If features and labels contain non float values then a ValueError is raisedand self._features and self._labels are None, None

If invalid data is passed to the constructors (i.e a list with different lengths) then self._features and self._labels are None, None

If invalid data is passed to the constructors (i.e a list that cannot be made into a homogenous array of floats) then self._features and self._labels are None, None

If a negative number is passed into NNData then train factor is 0

If a positive number is passed into NNData then train factor is 1

Process finished with exit code 0

"""