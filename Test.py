# from Assignment_Two import NNData
# from Assignment_Five import FFBPNetwork
# import collections
# import numpy as np
# import json
#
# # xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
# # xor_labels = [[0], [1], [1], [0]]
# # xor = NNData(xor_features, xor_labels, 0.5)
# # xor.prime_data()
# # print(xor.__dict__)
# # print()
# #
# # xor_dictionary = xor.__dict__.items()
# # for item in xor_dictionary:
# #     print(item)
# # print()
#
# # xor_1 = NNData()
# # print(xor_1.__dict__)
# # print(xor_1)
#
#
# class MultiTypeEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, NNData):
#             return{'__NNData__':o.__dict__}
#         elif isinstance(o, collections.deque):
#             return{'__deque__': list(o)}
#         elif isinstance(o, np.ndarray):
#             return{'__NDarray__' : o.tolist()}
#         else:
#             json.JSONEncoder.default(o)
#
#
# def multi_type_decoder(o):
#     if '__NNData__' in o:
#         # get NNData object
#         decode_object = o['__NNData__']
#
#         # get NNData object info
#         train_factor = decode_object['_train_factor']
#         train_indices = decode_object['_train_indices']
#         test_indices = decode_object['_test_indices']
#
#         train_pool = decode_object['_train_pool']['__deque__']
#         train_pool_deque = collections.deque(train_pool)
#
#         test_pool_list = decode_object['_test_pool']['__deque__']
#         test_pool_deque = collections.deque(test_pool_list)
#
#         features_list = decode_object['_features']['__NDarray__']
#         features_array = np.array(features_list)
#
#         labels_list = decode_object['_labels']['__NDarray__']
#         labels_array = np.array(labels_list)
#
#         ret_object = NNData()
#         # print(ret_object.__dict__)
#         ret_object._train_factor = train_factor
#         ret_object._train_indices = train_indices
#         ret_object._test_indices = test_indices
#         ret_object._train_pool = train_pool_deque
#         ret_object._test_pool = test_pool_deque
#         ret_object._features = features_array
#         ret_object._labels = features_list
#         # print(ret_object.__dict__)
#         return ret_object
#     else:
#         return o
#
#
# # with open('dat.txt', 'w') as f:
# #     json.dump(xor, f, cls=MultiTypeEncoder)
# #
# # with open('dat.txt', 'r') as f:
# #     my_obj = json.load(f, object_hook=multi_type_decoder)
# #     print(my_obj)
# #     print(type(my_obj))
# #     print(my_obj.__dict__)
#
#
# xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
# xor_labels = [[0], [1], [1], [0]]
# xor_data = NNData(xor_features, xor_labels)
#
# with open('xor_data.txt', 'w') as xor_data_encoded:
#     json.dump(xor_data, xor_data_encoded, cls=MultiTypeEncoder)
#
# with open('xor_data.txt', 'r') as xor_data_encoded:
#     xor_data_decoded = json.load(xor_data_encoded, object_hook=multi_type_decoder)
#
# print(xor_data_decoded.__dict__)
#
#
# def run_XOR():
#     """
#     This function uses the XOR data set to train and test the neural network.
#     """
#     network = FFBPNetwork(2, 1)
#     network.add_hidden_layer(3)
#
#     data = xor_data_decoded
#     network.train(data, 1000, order=NNData.Order.RANDOM)
# run_XOR()
#
# print('\nsine\n'
#       '----------------------')
#
# with open('sine_data.txt', 'r') as sin_encoded:
#     sin_decoded = json.load(sin_encoded, object_hook=multi_type_decoder)
#
#
# def run_sine():
#     network = FFBPNetwork(1, 1)
#     network.add_hidden_layer(3)
#
#     data = sin_decoded
#     network.train(data, 10000, order=NNData.Order.RANDOM)
#     network.test(data)
# run_sine()
#
#
#
#

