"""
Assignment Six: JSON
   Jonathan Fong
   8/6/2020
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
   Assignment 6: This program setups an encoding class called MultiTypeEncoder and a decoding function called multi-type
                 decoder. The program then tests to see if both the class and function are setup correctly.
"""
from Assignment_Two import NNData
from Assignment_Five import FFBPNetwork
import collections
import numpy as np
import json


class MultiTypeEncoder(json.JSONEncoder):
    """
    This class inherits from JSONEncoder and setups a multi-type encoder.
    """
    def default(self, o):
        """
        This function overrides the default method and adds new functionality to the default method.

        Parameters:
            o : any type of object
        """
        if isinstance(o, NNData):
            return {'__NNData__': o.__dict__}
        elif isinstance(o, collections.deque):
            return {'__deque__': list(o)}
        elif isinstance(o, np.ndarray):
            return {'__NDarray__': o.tolist()}
        else:
            json.JSONEncoder.default(o)


def multi_type_decoder(o):
    """
    This function decodes objects that are not natively supported by json.

    Parameters:
        o : any type of object
    """
    if '__NNData__' in o:
        # get NNData object
        decode_object = o['__NNData__']

        # get NNData object attributes
        train_factor = decode_object['_train_factor']

        train_indices = decode_object['_train_indices']
        test_indices = decode_object['_test_indices']

        train_pool_list = decode_object['_train_pool']['__deque__']
        train_pool_deque = collections.deque(train_pool_list)

        test_pool_list = decode_object['_test_pool']['__deque__']
        test_pool_deque = collections.deque(test_pool_list)

        features_list = decode_object['_features']['__NDarray__']
        features_array = np.array(features_list)

        labels_list = decode_object['_labels']['__NDarray__']
        labels_array = np.array(labels_list)

        # load NNData attributes
        ret_object = NNData()
        ret_object._train_factor = train_factor
        ret_object._train_indices = train_indices
        ret_object._test_indices = test_indices
        ret_object._train_pool = train_pool_deque
        ret_object._test_pool = test_pool_deque
        ret_object._features = features_array
        ret_object._labels = labels_array
        return ret_object
    else:
        return o


def main():
    """
    Runs program
    """
    print('Run XOR\n'
          '--------------------------------------------\n')
    # load XOR data
    xor_features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_labels = [[0], [1], [1], [0]]
    xor_data = NNData(xor_features, xor_labels)

    # encode XOR data
    with open('xor_data.txt', 'w') as xor_data_encoded:
        json.dump(xor_data, xor_data_encoded, cls=MultiTypeEncoder)

    # decode XOR data
    with open('xor_data.txt', 'r') as xor_data_encoded:
        json.load()
        xor_data_decoded = json.load(xor_data_encoded, object_hook=multi_type_decoder)

    def run_XOR():
        """
        This function uses the XOR data set to train and test the neural network.
        """
        network = FFBPNetwork(2, 1)
        network.add_hidden_layer(3)

        data = xor_data_decoded
        network.train(data, 10001, order=NNData.Order.RANDOM)
    run_XOR()

    print('Run Sine\n'
          '--------------------------------------------\n')
    # decode sine data
    with open('sine_data.txt', 'r') as sin_encoded:
        sin_decoded = json.load(sin_encoded, object_hook=multi_type_decoder)

    def run_sine():
        """
        This function uses the sine data set to train and test the neural network.
        """
        network = FFBPNetwork(1, 1)
        network.add_hidden_layer(3)

        data = sin_decoded
        network.train(data, 10001, order=NNData.Order.RANDOM)
        network.test(data)
    run_sine()


if __name__ == '__main__':
    main()

"""
"/Users/jonathanfong/Desktop/C S 3B (Summer)/venv/bin/python" "/Users/jonathanfong/Desktop/C S 3B (Summer)/AssignmentSix.py"
Run XOR
--------------------------------------------

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.839757054489722]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.8829661743153825]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.7838781551389543]

Epoch 0 root-mean-square error: 0.6879370195676837

Epoch 100 root-mean-square error: 0.5346945703574469

Epoch 200 root-mean-square error: 0.47683339764463306

Epoch 300 root-mean-square error: 0.4691195296672596

Epoch 400 root-mean-square error: 0.46621661186478836

Epoch 500 root-mean-square error: 0.4634523069065658

Epoch 600 root-mean-square error: 0.46021683148232145

Epoch 700 root-mean-square error: 0.45632786183954943

Epoch 800 root-mean-square error: 0.4516360171135896

Epoch 900 root-mean-square error: 0.4460094828938821

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.3379147348611508]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.38494608841005223]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.2941379934600332]

Epoch 1000 root-mean-square error: 0.4393156956734233

Epoch 1100 root-mean-square error: 0.4314463290326048

Epoch 1200 root-mean-square error: 0.42232366914881997

Epoch 1300 root-mean-square error: 0.4119093408847309

Epoch 1400 root-mean-square error: 0.40021634968081643

Epoch 1500 root-mean-square error: 0.3873111830704515

Epoch 1600 root-mean-square error: 0.37333649716937894

Epoch 1700 root-mean-square error: 0.35850831578738845

Epoch 1800 root-mean-square error: 0.343115074673254

Epoch 1900 root-mean-square error: 0.327491344498876

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.26638197687786913]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.2214049249276624]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.5852604131132008]

Epoch 2000 root-mean-square error: 0.31197672098978707

Epoch 2100 root-mean-square error: 0.2968763812692696

Epoch 2200 root-mean-square error: 0.2824296203447853

Epoch 2300 root-mean-square error: 0.2687993361632641

Epoch 2400 root-mean-square error: 0.2560754378567941

Epoch 2500 root-mean-square error: 0.24428638812232212

Epoch 2600 root-mean-square error: 0.23341850234977687

Epoch 2700 root-mean-square error: 0.22342848850238314

Epoch 2800 root-mean-square error: 0.2142577709277203

Epoch 2900 root-mean-square error: 0.20583863548163248

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.7530631254281872]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.20183373925737147]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.12657798394151917]

Epoch 3000 root-mean-square error: 0.19810490852458898

Epoch 3100 root-mean-square error: 0.19099007928574996

Epoch 3200 root-mean-square error: 0.18443328966614986

Epoch 3300 root-mean-square error: 0.17837824655210094

Epoch 3400 root-mean-square error: 0.17277414232046082

Epoch 3500 root-mean-square error: 0.16757542284988128

Epoch 3600 root-mean-square error: 0.16274190342902228

Epoch 3700 root-mean-square error: 0.15823766519160826

Epoch 3800 root-mean-square error: 0.15403067503576814

Epoch 3900 root-mean-square error: 0.15009295314409388

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.08336423526614593]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.8226776899603959]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.16095232764270567]

Epoch 4000 root-mean-square error: 0.14639950948562597

Epoch 4100 root-mean-square error: 0.14292818214884387

Epoch 4200 root-mean-square error: 0.13965928246223575

Epoch 4300 root-mean-square error: 0.13657542063351022

Epoch 4400 root-mean-square error: 0.13366093786730734

Epoch 4500 root-mean-square error: 0.13090186889636035

Epoch 4600 root-mean-square error: 0.12828568941239551

Epoch 4700 root-mean-square error: 0.1258013434879513

Epoch 4800 root-mean-square error: 0.12343865811792264

Epoch 4900 root-mean-square error: 0.12118852087776948

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.1355620491423535]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.857780162499842]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.06253011414814974]

Epoch 5000 root-mean-square error: 0.11904280255665166

Epoch 5100 root-mean-square error: 0.11699400540854406

Epoch 5200 root-mean-square error: 0.11503540478320255

Epoch 5300 root-mean-square error: 0.11316089788134548

Epoch 5400 root-mean-square error: 0.11136491717704691

Epoch 5500 root-mean-square error: 0.10964229662792692

Epoch 5600 root-mean-square error: 0.10798843673186909

Epoch 5700 root-mean-square error: 0.10639904847515562

Epoch 5800 root-mean-square error: 0.10487022928614063

Epoch 5900 root-mean-square error: 0.10339832379677663

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.11844606533650232]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.05046877714651112]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.8790734681561135]

Epoch 6000 root-mean-square error: 0.10198005354668409

Epoch 6100 root-mean-square error: 0.10061234068925468

Epoch 6200 root-mean-square error: 0.09929235496639419

Epoch 6300 root-mean-square error: 0.09801749561256358

Epoch 6400 root-mean-square error: 0.09678529747567265

Epoch 6500 root-mean-square error: 0.09559353303003416

Epoch 6600 root-mean-square error: 0.0944401010925937

Epoch 6700 root-mean-square error: 0.09332305872641658

Epoch 6800 root-mean-square error: 0.09224058417778021

Epoch 6900 root-mean-square error: 0.09119097982445697

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.8936249820632768]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.10606765336503166]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.04274738840005038]

Epoch 7000 root-mean-square error: 0.09017266907887973

Epoch 7100 root-mean-square error: 0.0891841752847705

Epoch 7200 root-mean-square error: 0.08822409921353545

Epoch 7300 root-mean-square error: 0.08729115961980315

Epoch 7400 root-mean-square error: 0.08638411719044647

Epoch 7500 root-mean-square error: 0.08550183466940879

Epoch 7600 root-mean-square error: 0.08464323445054762

Epoch 7700 root-mean-square error: 0.08380730530120564

Epoch 7800 root-mean-square error: 0.08299308546070391

Epoch 7900 root-mean-square error: 0.0821996814774343

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.03729184591314961]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.09654597376693544]

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.9041934602359181]

Epoch 8000 root-mean-square error: 0.0814262445074938

Epoch 8100 root-mean-square error: 0.08067195924546869

Epoch 8200 root-mean-square error: 0.07993608175200695

Epoch 8300 root-mean-square error: 0.07921787751500187

Epoch 8400 root-mean-square error: 0.07851668047372982

Epoch 8500 root-mean-square error: 0.07783184085414933

Epoch 8600 root-mean-square error: 0.07716274292008905

Epoch 8700 root-mean-square error: 0.07650880908877393

Epoch 8800 root-mean-square error: 0.07586949064898568

Epoch 8900 root-mean-square error: 0.07524425906673753

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.9124021596165517]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.0890539932108755]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.033257874935171725]

Epoch 9000 root-mean-square error: 0.07463261483616546

Epoch 9100 root-mean-square error: 0.0740340893603297

Epoch 9200 root-mean-square error: 0.07344822594782738

Epoch 9300 root-mean-square error: 0.07287459505215858

Epoch 9400 root-mean-square error: 0.07231278687953047

Epoch 9500 root-mean-square error: 0.0717624071965593

Epoch 9600 root-mean-square error: 0.07122308402168286

Epoch 9700 root-mean-square error: 0.07069445761116326

Epoch 9800 root-mean-square error: 0.07017618970673821

Epoch 9900 root-mean-square error: 0.06966795097964722

Training Information
-------------------------
input values: [0. 1.]
expected values: [1.]
output values: [0.9189370998959823]

Training Information
-------------------------
input values: [1. 1.]
expected values: [0.]
output values: [0.0301463202677648]

Training Information
-------------------------
input values: [0. 0.]
expected values: [0.]
output values: [0.08290497891207915]

Epoch 10000 root-mean-square error: 0.06916942948869077

Final root-mean-square error: 0.06916942948869077

Run Sine
--------------------------------------------

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.6630729406600878]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.6636244468422595]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.6729079363990248]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.6545182780946863]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.6514827837885002]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.6397987158021615]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.6545703613460038]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6573199614431021]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.6670682088267307]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6566156553963522]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.64413397298818]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.6492754490965391]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.6602720034204489]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.6806439680123364]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.6387459047622494]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.6661605706786597]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.6299727960885256]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.6440253594952752]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.6517609890331969]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.6784130986337522]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.6421255353335806]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.6468287208871993]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.6756432960350217]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.6766975933732875]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.6349897036110483]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.6466142633710684]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.6785491264981804]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.6459052270129741]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.6488027944243668]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.6306298227368901]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.6576047078557804]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.6454384347822227]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.632028017494643]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.634530537351692]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.6336101447083815]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.6440957141918517]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.6296285124024882]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.643350338053857]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.6247940531582052]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.6533914817927469]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.6410247765438218]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.6362476070483193]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6446652798038218]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.6526460626455213]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.6406944564234583]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.6367055248962155]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.6498105022821633]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.6585663503819646]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.6697866625787519]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.6704839508775985]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.6645694630763874]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.6528128777190076]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.6381410943179379]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.6546553476618091]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.6408146786738755]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.648307135072139]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.6287979457433492]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.6634349104736345]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.6719567426129115]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.6230734811909675]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.6536513785388501]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.6376206383700476]

Epoch 0 root-mean-square error: 0.2831694826034222

Epoch 100 root-mean-square error: 0.2700233124071557

Epoch 200 root-mean-square error: 0.2568543559482531

Epoch 300 root-mean-square error: 0.22226907017290357

Epoch 400 root-mean-square error: 0.1798293225230938

Epoch 500 root-mean-square error: 0.1477098027878671

Epoch 600 root-mean-square error: 0.12508430660860703

Epoch 700 root-mean-square error: 0.10864823581518214

Epoch 800 root-mean-square error: 0.09623024534078856

Epoch 900 root-mean-square error: 0.08653175310383558

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.7930712826922093]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.760080463586579]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.190533379831524]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.49132945736861255]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7371881207156227]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.280185502485293]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.8448687441882996]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.6975487578052081]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5373801488042249]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.42661354978227195]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6417428099144261]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.4429083937947543]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5927013324384782]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7100716344717694]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6241005617950487]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8123027297209322]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.3034827181724302]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.5067769735048641]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.23604354931495033]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4986830399488297]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.2500388030276094]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.846303059580593]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.543995499894794]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.8345111765382426]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5585891537942491]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7058438886348712]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.38474200893414656]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4669517497563968]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7258214626413545]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.1510945523352689]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.843753528299257]

Epoch 1000 root-mean-square error: 0.07876596433404252

Epoch 1100 root-mean-square error: 0.07242855615490384

Epoch 1200 root-mean-square error: 0.06717729677271432

Epoch 1300 root-mean-square error: 0.06278019688226301

Epoch 1400 root-mean-square error: 0.059057468227228564

Epoch 1500 root-mean-square error: 0.055882347110435014

Epoch 1600 root-mean-square error: 0.05315502999274723

Epoch 1700 root-mean-square error: 0.050797604914736914

Epoch 1800 root-mean-square error: 0.04874879307747752

Epoch 1900 root-mean-square error: 0.04695943173014726

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7407026187895631]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.38658365459954197]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.8823218007290556]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8586457337275156]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6563033501883752]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7357965929449368]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.725126224758124]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.8947376250719581]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5240734215521297]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.53324298524135]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.2454299747641195]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.48583864736990123]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.2205623216729182]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4365700786846935]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4661903486494831]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.891921985067998]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.13224264799301874]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.8929092737909774]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.1900247567509157]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6342259863456793]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8376061776505913]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.7999779785258986]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.594711295015389]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.1758311840764898]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7732139676018259]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5513793966372119]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.09785196205357499]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.40661985516087323]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.3372575275139081]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4759656942985688]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7599026879346281]

Epoch 2000 root-mean-square error: 0.04538969315853315

Epoch 2100 root-mean-square error: 0.04400582101827414

Epoch 2200 root-mean-square error: 0.042780730774998

Epoch 2300 root-mean-square error: 0.041691414029358206

Epoch 2400 root-mean-square error: 0.040720161629246784

Epoch 2500 root-mean-square error: 0.03984867277849379

Epoch 2600 root-mean-square error: 0.03906739290200634

Epoch 2700 root-mean-square error: 0.03836141757772661

Epoch 2800 root-mean-square error: 0.03772196523940787

Epoch 2900 root-mean-square error: 0.03714121395278775

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.32193428020227305]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.20459307619012218]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7455408995279534]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6358011813512783]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.17456360251473166]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9007879124115671]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.37283517965423524]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8762640170064578]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9116809970840867]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5275170415876332]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7859255462530866]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8144405008279715]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5933463436157571]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9107274202623785]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22929790327789198]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11924736283728575]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7719041091801736]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.3938345333387555]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4251978537873408]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.45646384939270623]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7512491400568018]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.160937065240113]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5174840752165588]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.546765726828772]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4665737485733907]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.0868773941363715]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9134467761491021]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.47692308497715424]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6596240877974006]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8541958415696183]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7340702834737589]

Epoch 3000 root-mean-square error: 0.03661145924341892

Epoch 3100 root-mean-square error: 0.0361273252002358

Epoch 3200 root-mean-square error: 0.03568288945953768

Epoch 3300 root-mean-square error: 0.03527431557226272

Epoch 3400 root-mean-square error: 0.034896912881544694

Epoch 3500 root-mean-square error: 0.034548171417834

Epoch 3600 root-mean-square error: 0.03422434050448475

Epoch 3700 root-mean-square error: 0.033923008668857356

Epoch 3800 root-mean-square error: 0.033642356184919835

Epoch 3900 root-mean-square error: 0.033379950421048084

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7768109685789936]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.38776248548591447]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6602118425670989]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7374899250434873]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5433838520409119]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22347257111998692]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8851882078169503]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8624927328062926]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.1993963909959277]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9215570734227121]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.635705098691005]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.41945042696739027]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5135339116665084]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5913622832328862]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.1157038571783108]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08426198148257853]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8209684957697378]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.45095676235528553]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.31567095820045765]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.1699066490529949]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7494336724829667]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.47195901924868844]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9104793318390146]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7913645964980518]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9205484245352973]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15647590191926594]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.3668426654912417]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7553373797085264]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5236052801635717]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4616053981845502]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9234091482715643]

Epoch 4000 root-mean-square error: 0.03313419363344629

Epoch 4100 root-mean-square error: 0.032903782990105684

Epoch 4200 root-mean-square error: 0.03268689538601013

Epoch 4300 root-mean-square error: 0.03248269580981325

Epoch 4400 root-mean-square error: 0.03228970287623692

Epoch 4500 root-mean-square error: 0.03210802238229461

Epoch 4600 root-mean-square error: 0.03193551743764437

Epoch 4700 root-mean-square error: 0.031772269382106345

Epoch 4800 root-mean-square error: 0.03161682158992265

Epoch 4900 root-mean-square error: 0.031469626459550044

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.867483700216281]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.19785720094123482]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7515561966426454]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5111356482975754]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7573320464381056]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7390863123062272]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7792284842093607]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8906281756261109]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22165547981734787]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5212037150891167]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9266852838460964]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8248979354668668]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.41650094417624056]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6348978914480672]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5898034788422717]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9295582111259197]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.3847645746182749]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4588723075204084]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6599532137485679]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9276757172316615]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11519106966343991]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4482501284386455]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15540545966763664]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16870591705721896]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.794371632748757]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5413668627794249]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.4692693925720993]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9164783793179453]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.36391669842113666]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08412894451192289]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.3131171656077475]

Epoch 5000 root-mean-square error: 0.031329164676365505

Epoch 5100 root-mean-square error: 0.031195564450434785

Epoch 5200 root-mean-square error: 0.031067978990323267

Epoch 5300 root-mean-square error: 0.030946054441301527

Epoch 5400 root-mean-square error: 0.030829093336440905

Epoch 5500 root-mean-square error: 0.030718033656505998

Epoch 5600 root-mean-square error: 0.03061126990733703

Epoch 5700 root-mean-square error: 0.030508642500086856

Epoch 5800 root-mean-square error: 0.03041001866485158

Epoch 5900 root-mean-square error: 0.030316033218459205

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5399765757650017]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16859986978141245]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9205918227702583]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15545071324494522]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7809571314674747]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6341394748199691]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6594048296906522]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.19730247187955016]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.3116299722427076]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.36221850522873406]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.758358304323386]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9338033772283814]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7961642916696892]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7523004852869253]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22102101021681644]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5092154691680973]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.456907428856026]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9308768525212318]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5886556575339418]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8271939678668807]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.44636108536464736]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.41463139021614337]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11552181497987338]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7398188036085744]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8705502757241643]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08462551153260417]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.3829514871529391]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5195143959907954]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9318761036982894]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.46757943297820426]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8943832116325032]

Epoch 6000 root-mean-square error: 0.030225224072921764

Epoch 6100 root-mean-square error: 0.03013802049160561

Epoch 6200 root-mean-square error: 0.03005362449644817

Epoch 6300 root-mean-square error: 0.02997309319972304

Epoch 6400 root-mean-square error: 0.029895222247983702

Epoch 6500 root-mean-square error: 0.02981979896030456

Epoch 6600 root-mean-square error: 0.029747385585276294

Epoch 6700 root-mean-square error: 0.02967669938002113

Epoch 6800 root-mean-square error: 0.029609559705147035

Epoch 6900 root-mean-square error: 0.029544062164838646

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9369520081672075]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7591765075478539]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6590811649730012]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.538616156049817]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.3818799693932945]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5876476894048954]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8969718233440594]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.3612308495613391]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16884767610187001]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8729994507130856]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08531358418454302]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.740356894234588]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.31094296871820887]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4557950165792903]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6334771203324013]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.828886982888332]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7817481206080543]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4450362132833586]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7528296730376169]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11601545291571405]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9349116479132191]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7972182754044163]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9339129174691204]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.19735471570445035]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22089098175914057]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15570618961850893]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5080026531047211]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4134008680602842]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5183483235218483]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.46626256646747766]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9235488135861478]

Epoch 7000 root-mean-square error: 0.029480791346889097

Epoch 7100 root-mean-square error: 0.029419092359004747

Epoch 7200 root-mean-square error: 0.02936022037517223

Epoch 7300 root-mean-square error: 0.029302678257023852

Epoch 7400 root-mean-square error: 0.029246638439673095

Epoch 7500 root-mean-square error: 0.02919308518279708

Epoch 7600 root-mean-square error: 0.02914045791482524

Epoch 7700 root-mean-square error: 0.029089721895967438

Epoch 7800 root-mean-square error: 0.02904047688711696

Epoch 7900 root-mean-square error: 0.02899252863662164

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.3812649833796871]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.9364038425952163]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5176866051941065]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7407854529115553]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.19764273658368517]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.1166723205278472]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6586996284495158]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9373568610486248]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22100536511571667]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5072122398371303]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7534070776685047]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7594394196095927]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.8990046901074857]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.782457969866982]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.3604238349963714]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9258318345170388]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.586955841176799]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.41257123065786877]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15614857653025072]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5377942000761303]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6328117076286327]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08593113944531967]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.8746263813044306]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.4651898171852414]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.31042879421814196]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7982242838965451]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4441667615266937]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16914397901486908]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8301501350179025]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.45474697498869476]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9393179310612796]

Epoch 8000 root-mean-square error: 0.028945833442113336

Epoch 8100 root-mean-square error: 0.02890068496834838

Epoch 8200 root-mean-square error: 0.028856344194801684

Epoch 8300 root-mean-square error: 0.028813357705385165

Epoch 8400 root-mean-square error: 0.028772021658380706

Epoch 8500 root-mean-square error: 0.028731429138918806

Epoch 8600 root-mean-square error: 0.028691877218683885

Epoch 8700 root-mean-square error: 0.028652764470031452

Epoch 8800 root-mean-square error: 0.02861587940389396

Epoch 8900 root-mean-square error: 0.02857909404035036

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9277149907994967]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7599679180332468]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.45410565392151336]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.7408334303768682]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7535750461682458]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.876059271683337]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6322270293436608]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.31002495135840125]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5061946338087739]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.9006474921960291]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16949806364558528]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.9392616911732653]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.1565569193302336]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22110999381854218]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.7989766360272129]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.831156634551088]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5862214138840055]

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.197775184788131]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5165501784167302]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7829554590618523]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.938198365221893]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5369093510191368]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.38049904647443844]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4434809461835665]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6582581785860708]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.941199912710656]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.360011527625339]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08656227661655871]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.46459129452558684]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4119776576711772]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11721617220065321]

Epoch 9000 root-mean-square error: 0.028543533809705302

Epoch 9100 root-mean-square error: 0.028508948630911947

Epoch 9200 root-mean-square error: 0.028475055291886275

Epoch 9300 root-mean-square error: 0.028441953731585132

Epoch 9400 root-mean-square error: 0.02840848080116755

Epoch 9500 root-mean-square error: 0.02837810652688884

Epoch 9600 root-mean-square error: 0.028346325867321446

Epoch 9700 root-mean-square error: 0.028316921852707972

Epoch 9800 root-mean-square error: 0.028287631192005013

Epoch 9900 root-mean-square error: 0.028258735743389638

Training Information
-------------------------
input values: [0.21]
expected values: [0.2084599]
output values: [0.19805336191495623]

Training Information
-------------------------
input values: [0.53]
expected values: [0.50553334]
output values: [0.5057533335896809]

Training Information
-------------------------
input values: [0.61]
expected values: [0.57286746]
output values: [0.5857576391087933]

Training Information
-------------------------
input values: [0.66]
expected values: [0.61311685]
output values: [0.6318962979857724]

Training Information
-------------------------
input values: [1.46]
expected values: [0.99386836]
output values: [0.940797684438038]

Training Information
-------------------------
input values: [0.54]
expected values: [0.51413599]
output values: [0.5159306906986907]

Training Information
-------------------------
input values: [0.17]
expected values: [0.16918235]
output values: [0.16982024595305686]

Training Information
-------------------------
input values: [0.48]
expected values: [0.46177918]
output values: [0.4533661309162675]

Training Information
-------------------------
input values: [1.45]
expected values: [0.99271299]
output values: [0.939812229064312]

Training Information
-------------------------
input values: [0.69]
expected values: [0.63653718]
output values: [0.6580025173271904]

Training Information
-------------------------
input values: [0.56]
expected values: [0.5311862]
output values: [0.5363120876185972]

Training Information
-------------------------
input values: [0.15]
expected values: [0.14943813]
output values: [0.15688771004459032]

Training Information
-------------------------
input values: [1.1]
expected values: [0.89120736]
output values: [0.877248709203724]

Training Information
-------------------------
input values: [0.83]
expected values: [0.73793137]
output values: [0.7599989029879948]

Training Information
-------------------------
input values: [0.24]
expected values: [0.23770263]
output values: [0.22112466837741457]

Training Information
-------------------------
input values: [0.87]
expected values: [0.76432894]
output values: [0.7833401824707216]

Training Information
-------------------------
input values: [0.47]
expected values: [0.45288629]
output values: [0.4426768310031674]

Training Information
-------------------------
input values: [0.34]
expected values: [0.33348709]
output values: [0.30983225891645666]

Training Information
-------------------------
input values: [0.49]
expected values: [0.47062589]
output values: [0.46387790681002833]

Training Information
-------------------------
input values: [0.01]
expected values: [0.00999983]
output values: [0.08707561311837732]

Training Information
-------------------------
input values: [0.08]
expected values: [0.07991469]
output values: [0.11765461692275483]

Training Information
-------------------------
input values: [0.82]
expected values: [0.73114583]
output values: [0.7536929571955754]

Training Information
-------------------------
input values: [0.39]
expected values: [0.38018842]
output values: [0.3593894782695866]

Training Information
-------------------------
input values: [1.48]
expected values: [0.99588084]
output values: [0.9427621461507651]

Training Information
-------------------------
input values: [0.9]
expected values: [0.78332691]
output values: [0.799392711001782]

Training Information
-------------------------
input values: [0.44]
expected values: [0.42593947]
output values: [0.4112376447434143]

Training Information
-------------------------
input values: [1.36]
expected values: [0.9778646]
output values: [0.9291833109695983]

Training Information
-------------------------
input values: [1.2]
expected values: [0.93203909]
output values: [0.9020772228836174]

Training Information
-------------------------
input values: [0.97]
expected values: [0.82488571]
output values: [0.8320415517580709]

Training Information
-------------------------
input values: [0.8]
expected values: [0.71735609]
output values: [0.740949577997733]

Training Information
-------------------------
input values: [0.41]
expected values: [0.39860933]
output values: [0.38009645440204964]

Epoch 10000 root-mean-square error: 0.02823087587349236

Final root-mean-square error: 0.02823087587349236

Testing Information
------------------------
input values: [0.22]
expected values: [0.21822962]
ouput values: [0.20559584547908022]

Testing Information
------------------------
input values: [0.73]
expected values: [0.66686964]
ouput values: [0.6904611882487408]

Testing Information
------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.28161162346157603]

Testing Information
------------------------
input values: [1.06]
expected values: [0.87235548]
ouput values: [0.8651762085507576]

Testing Information
------------------------
input values: [0.12]
expected values: [0.11971221]
ouput values: [0.13899757014687275]

Testing Information
------------------------
input values: [1.21]
expected values: [0.935616]
ouput values: [0.9041649846400014]

Testing Information
------------------------
input values: [1.12]
expected values: [0.90010044]
ouput values: [0.8829317799826493]

Testing Information
------------------------
input values: [0.14]
expected values: [0.13954311]
ouput values: [0.15079738097508869]

Testing Information
------------------------
input values: [0.1]
expected values: [0.09983342]
ouput values: [0.12799352158342725]

Testing Information
------------------------
input values: [0.57]
expected values: [0.53963205]
ouput values: [0.5465700462251744]

Testing Information
------------------------
input values: [1.47]
expected values: [0.99492435]
ouput values: [0.9418569338076095]

Testing Information
------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9172711392607019]

Testing Information
------------------------
input values: [0.29]
expected values: [0.28595223]
ouput values: [0.26368222643323397]

Testing Information
------------------------
input values: [0.38]
expected values: [0.37092047]
ouput values: [0.34975296029755953]

Testing Information
------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.1448890602500088]

Testing Information
------------------------
input values: [1.49]
expected values: [0.99673775]
ouput values: [0.9438650504986628]

Testing Information
------------------------
input values: [0.36]
expected values: [0.35227423]
ouput values: [0.3298796238481742]

Testing Information
------------------------
input values: [1.19]
expected values: [0.92836897]
ouput values: [0.9001986264234652]

Testing Information
------------------------
input values: [1.25]
expected values: [0.94898462]
ouput values: [0.9122371001723176]

Testing Information
------------------------
input values: [0.65]
expected values: [0.60518641]
ouput values: [0.6238199651338202]

Testing Information
------------------------
input values: [0.74]
expected values: [0.67428791]
ouput values: [0.6987679217752751]

Testing Information
------------------------
input values: [0.02]
expected values: [0.01999867]
ouput values: [0.091026271933001]

Testing Information
------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7893326852718737]

Testing Information
------------------------
input values: [0.95]
expected values: [0.8134155]
ouput values: [0.8235902230234827]

Testing Information
------------------------
input values: [1.53]
expected values: [0.99916795]
ouput values: [0.9473585196330505]

Testing Information
------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.10362230414558145]

Testing Information
------------------------
input values: [0.51]
expected values: [0.48817725]
ouput values: [0.4852406940956998]

Testing Information
------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.5955939103785235]

Testing Information
------------------------
input values: [1.15]
expected values: [0.91276394]
ouput values: [0.8907451758291903]

Testing Information
------------------------
input values: [0.3]
expected values: [0.29552021]
ouput values: [0.27269791775817415]

Testing Information
------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6749121441242516]

Testing Information
------------------------
input values: [0.]
expected values: [0.]
ouput values: [0.08334870473086506]

Testing Information
------------------------
input values: [1.35]
expected values: [0.97572336]
ouput values: [0.9279310669060927]

Testing Information
------------------------
input values: [0.58]
expected values: [0.54802394]
ouput values: [0.5567562488989516]

Testing Information
------------------------
input values: [0.37]
expected values: [0.36161543]
ouput values: [0.3396067414214568]

Testing Information
------------------------
input values: [1.16]
expected values: [0.91680311]
ouput values: [0.8932254466535363]

Testing Information
------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8364089480065962]

Testing Information
------------------------
input values: [0.19]
expected values: [0.18885889]
ouput values: [0.18367803915121625]

Testing Information
------------------------
input values: [0.33]
expected values: [0.32404303]
ouput values: [0.30064945380319463]

Testing Information
------------------------
input values: [1.17]
expected values: [0.9207506]
ouput values: [0.89565872189053]

Testing Information
------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9498341346807501]

Testing Information
------------------------
input values: [0.91]
expected values: [0.78950374]
ouput values: [0.8050492621569347]

Testing Information
------------------------
input values: [1.54]
expected values: [0.99952583]
ouput values: [0.9482533721982466]

Testing Information
------------------------
input values: [1.08]
expected values: [0.88195781]
ouput values: [0.8718259169460167]

Testing Information
------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9448605047837393]

Testing Information
------------------------
input values: [1.]
expected values: [0.84147098]
ouput values: [0.8445668123279286]

Testing Information
------------------------
input values: [1.31]
expected values: [0.96618495]
ouput values: [0.9223353738533102]

Testing Information
------------------------
input values: [0.28]
expected values: [0.27635565]
ouput values: [0.255209753796358]

Testing Information
------------------------
input values: [0.55]
expected values: [0.52268723]
ouput values: [0.527248364293829]

Testing Information
------------------------
input values: [1.14]
expected values: [0.9086335]
ouput values: [0.8886020322897189]

Testing Information
------------------------
input values: [0.99]
expected values: [0.83602598]
ouput values: [0.8408151720532128]

Testing Information
------------------------
input values: [1.39]
expected values: [0.98370081]
ouput values: [0.9333669129127301]

Testing Information
------------------------
input values: [0.89]
expected values: [0.77707175]
ouput values: [0.7951776957195543]

Testing Information
------------------------
input values: [0.84]
expected values: [0.74464312]
ouput values: [0.7670391738611926]

Testing Information
------------------------
input values: [0.03]
expected values: [0.0299955]
ouput values: [0.09510621208121899]

Testing Information
------------------------
input values: [0.23]
expected values: [0.22797752]
ouput values: [0.21364320273777573]

Testing Information
------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8688550664372562]

Testing Information
------------------------
input values: [1.29]
expected values: [0.96083506]
ouput values: [0.9192121956991041]

Testing Information
------------------------
input values: [0.96]
expected values: [0.81919157]
ouput values: [0.8284298617577243]

Testing Information
------------------------
input values: [0.72]
expected values: [0.65938467]
ouput values: [0.6834795993585285]

Testing Information
------------------------
input values: [0.79]
expected values: [0.71035327]
ouput values: [0.7349023989352798]

Testing Information
------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.133501368853212]

Testing Information
------------------------
input values: [1.57]
expected values: [0.99999968]
ouput values: [0.9505940930472023]

Testing Information
------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9367836451581852]

Testing Information
------------------------
input values: [1.4]
expected values: [0.98544973]
ouput values: [0.9345042576522479]

Testing Information
------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.614781439232017]

Testing Information
------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.9140656019003935]

Testing Information
------------------------
input values: [1.01]
expected values: [0.84683184]
ouput values: [0.8483805320232095]

Testing Information
------------------------
input values: [0.75]
expected values: [0.68163876]
ouput values: [0.7064760917826288]

Testing Information
------------------------
input values: [1.41]
expected values: [0.9871001]
ouput values: [0.9356501306903797]

Testing Information
------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9491020402455441]

Testing Information
------------------------
input values: [1.04]
expected values: [0.86240423]
ouput values: [0.8590816336665063]

Testing Information
------------------------
input values: [0.43]
expected values: [0.4168708]
ouput values: [0.4016888120440629]

Testing Information
------------------------
input values: [0.94]
expected values: [0.8075581]
ouput values: [0.8195533694726977]

Testing Information
------------------------
input values: [0.76]
expected values: [0.68892145]
ouput values: [0.7139052349420075]

Testing Information
------------------------
input values: [0.67]
expected values: [0.62098599]
ouput values: [0.641540754546856]

Testing Information
------------------------
input values: [1.02]
expected values: [0.85210802]
ouput values: [0.8518914745902723]

Testing Information
------------------------
input values: [1.13]
expected values: [0.90441219]
ouput values: [0.8858623316243184]

Testing Information
------------------------
input values: [0.46]
expected values: [0.44394811]
ouput values: [0.4330099830679908]

Testing Information
------------------------
input values: [1.37]
expected values: [0.97990806]
ouput values: [0.9307780410790635]

Testing Information
------------------------
input values: [0.4]
expected values: [0.38941834]
ouput values: [0.3705293093595767]

Testing Information
------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.11295112067020423]

Testing Information
------------------------
input values: [0.06]
expected values: [0.05996401]
ouput values: [0.10822221597411127]

Testing Information
------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.925339522006844]

Testing Information
------------------------
input values: [0.32]
expected values: [0.31456656]
ouput values: [0.2915288637543812]

Testing Information
------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9380041242820937]

Testing Information
------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.4229141448372959]

Testing Information
------------------------
input values: [0.78]
expected values: [0.70327942]
ouput values: [0.7284432003740996]

Testing Information
------------------------
input values: [1.18]
expected values: [0.92460601]
ouput values: [0.8982030297661732]

Testing Information
------------------------
input values: [0.86]
expected values: [0.75784256]
ouput values: [0.7788432593373202]

Testing Information
------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.2381091109382669]

Testing Information
------------------------
input values: [1.23]
expected values: [0.9424888]
ouput values: [0.9086460722720451]

Testing Information
------------------------
input values: [0.68]
expected values: [0.62879302]
ouput values: [0.6506585416108387]

Testing Information
------------------------
input values: [0.18]
expected values: [0.17902957]
ouput values: [0.17693512797966607]

Testing Information
------------------------
input values: [0.81]
expected values: [0.72428717]
ouput values: [0.7483997633678553]

Testing Information
------------------------
input values: [1.51]
expected values: [0.99815247]
ouput values: [0.9458005997160152]

Testing Information
------------------------
input values: [1.38]
expected values: [0.98185353]
ouput values: [0.9321318958923994]

Testing Information
------------------------
input values: [0.85]
expected values: [0.75128041]
ouput values: [0.773033849537206]

Testing Information
------------------------
input values: [1.27]
expected values: [0.95510086]
ouput values: [0.9159007650525641]

Testing Information
------------------------
input values: [1.09]
expected values: [0.88662691]
ouput values: [0.8750332709492317]

Testing Information
------------------------
input values: [1.05]
expected values: [0.86742323]
ouput values: [0.8625889673855346]

Testing Information
------------------------
input values: [0.2]
expected values: [0.19866933]
ouput values: [0.19110942472688858]

Testing Information
------------------------
input values: [1.24]
expected values: [0.945784]
ouput values: [0.9105776745041266]

Testing Information
------------------------
input values: [1.3]
expected values: [0.96355819]
ouput values: [0.9209524454158085]

Testing Information
------------------------
input values: [0.52]
expected values: [0.49688014]
ouput values: [0.4966798164791174]

Testing Information
------------------------
input values: [0.7]
expected values: [0.64421769]
ouput values: [0.6676779001372533]

Testing Information
------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9067265676925432]

Testing Information
------------------------
input values: [1.03]
expected values: [0.85729899]
ouput values: [0.8558881531486621]

Testing Information
------------------------
input values: [0.27]
expected values: [0.26673144]
ouput values: [0.2467449844774082]

Testing Information
------------------------
input values: [0.35]
expected values: [0.34289781]
ouput values: [0.320624578309055]

Testing Information
------------------------
input values: [0.42]
expected values: [0.40776045]
ouput values: [0.39176545075986136]

Testing Information
------------------------
input values: [1.11]
expected values: [0.89569869]
ouput values: [0.8809977044570754]

Testing Information
------------------------
input values: [0.25]
expected values: [0.24740396]
ouput values: [0.2301021043772728]

Testing Information
------------------------
input values: [0.04]
expected values: [0.03998933]
ouput values: [0.099433250416389]

Testing Information
------------------------
input values: [1.34]
expected values: [0.97348454]
ouput values: [0.9270320219614284]

Testing Information
------------------------
input values: [1.52]
expected values: [0.99871014]
ouput values: [0.9468701124440766]

Testing Information
------------------------
input values: [0.77]
expected values: [0.69613524]
ouput values: [0.7219623699842046]

Testing Information
------------------------
input values: [0.63]
expected values: [0.58914476]
ouput values: [0.6063306815731692]

Testing Information
------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5678953148099174]

Testing Information
------------------------
input values: [0.92]
expected values: [0.79560162]
ouput values: [0.8105342726222431]

Testing Information
------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9391716059708151]

Testing Information
------------------------
input values: [0.09]
expected values: [0.08987855]
ouput values: [0.12300310002212632]

Testing Information
------------------------
input values: [1.32]
expected values: [0.9687151]
ouput values: [0.9240770769492834]

Testing Information
------------------------
input values: [0.5]
expected values: [0.47942554]
ouput values: [0.4759332272359744]

Testing Information
------------------------
input values: [0.93]
expected values: [0.80161994]
ouput values: [0.8153662239543203]

Testing Information
------------------------
input values: [0.6]
expected values: [0.56464247]
ouput values: [0.577609226452993]

Testing Information
------------------------
input values: [0.16]
expected values: [0.15931821]
ouput values: [0.16368604969806858]

Testing Information
------------------------
input values: [0.]
expected values: [0.]
ouput values: [0.0834510744438245]

Testing Information
------------------------
input values: [0.02]
expected values: [0.01999867]
ouput values: [0.09108158246681279]

Testing Information
------------------------
input values: [0.03]
expected values: [0.0299955]
ouput values: [0.09511774877677831]

Testing Information
------------------------
input values: [0.04]
expected values: [0.03998933]
ouput values: [0.09931255143135924]

Testing Information
------------------------
input values: [0.05]
expected values: [0.04997917]
ouput values: [0.10367000837316419]

Testing Information
------------------------
input values: [0.06]
expected values: [0.05996401]
ouput values: [0.10819404679460994]

Testing Information
------------------------
input values: [0.07]
expected values: [0.06994285]
ouput values: [0.11288848419388316]

Testing Information
------------------------
input values: [0.09]
expected values: [0.08987855]
ouput values: [0.12282046123585612]

Testing Information
------------------------
input values: [0.1]
expected values: [0.09983342]
ouput values: [0.12804835343008364]

Testing Information
------------------------
input values: [0.11]
expected values: [0.1097783]
ouput values: [0.13346042467583147]

Testing Information
------------------------
input values: [0.12]
expected values: [0.11971221]
ouput values: [0.13905962458805554]

Testing Information
------------------------
input values: [0.13]
expected values: [0.12963414]
ouput values: [0.14484865880742168]

Testing Information
------------------------
input values: [0.14]
expected values: [0.13954311]
ouput values: [0.15082996349864844]

Testing Information
------------------------
input values: [0.16]
expected values: [0.15931821]
ouput values: [0.16338403699960927]

Testing Information
------------------------
input values: [0.18]
expected values: [0.17902957]
ouput values: [0.1767234291632804]

Testing Information
------------------------
input values: [0.19]
expected values: [0.18885889]
ouput values: [0.1836916392419981]

Testing Information
------------------------
input values: [0.2]
expected values: [0.19866933]
ouput values: [0.19085957351655342]

Testing Information
------------------------
input values: [0.22]
expected values: [0.21822962]
ouput values: [0.2057789037855956]

Testing Information
------------------------
input values: [0.23]
expected values: [0.22797752]
ouput values: [0.2135421930076404]

Testing Information
------------------------
input values: [0.25]
expected values: [0.24740396]
ouput values: [0.2296251135281327]

Testing Information
------------------------
input values: [0.26]
expected values: [0.25708055]
ouput values: [0.2379672854339421]

Testing Information
------------------------
input values: [0.27]
expected values: [0.26673144]
ouput values: [0.2464965318287358]

Testing Information
------------------------
input values: [0.28]
expected values: [0.27635565]
ouput values: [0.2552087169690472]

Testing Information
------------------------
input values: [0.29]
expected values: [0.28595223]
ouput values: [0.2640990705326463]

Testing Information
------------------------
input values: [0.3]
expected values: [0.29552021]
ouput values: [0.2731621791894668]

Testing Information
------------------------
input values: [0.31]
expected values: [0.30505864]
ouput values: [0.2823919828330687]

Testing Information
------------------------
input values: [0.32]
expected values: [0.31456656]
ouput values: [0.2917817759793433]

Testing Information
------------------------
input values: [0.33]
expected values: [0.32404303]
ouput values: [0.30132421479463267]

Testing Information
------------------------
input values: [0.35]
expected values: [0.34289781]
ouput values: [0.3207560310058275]

Testing Information
------------------------
input values: [0.36]
expected values: [0.35227423]
ouput values: [0.33070428731660967]

Testing Information
------------------------
input values: [0.37]
expected values: [0.36161543]
ouput values: [0.34076985957932526]

Testing Information
------------------------
input values: [0.38]
expected values: [0.37092047]
ouput values: [0.35094254460206303]

Testing Information
------------------------
input values: [0.4]
expected values: [0.38941834]
ouput values: [0.3714802128466574]

Testing Information
------------------------
input values: [0.42]
expected values: [0.40776045]
ouput values: [0.3923137683760186]

Testing Information
------------------------
input values: [0.43]
expected values: [0.4168708]
ouput values: [0.4028521630861898]

Testing Information
------------------------
input values: [0.45]
expected values: [0.43496553]
ouput values: [0.42396242639359355]

Testing Information
------------------------
input values: [0.46]
expected values: [0.44394811]
ouput values: [0.4345785874428686]

Testing Information
------------------------
input values: [0.5]
expected values: [0.47942554]
ouput values: [0.4768299381599211]

Testing Information
------------------------
input values: [0.51]
expected values: [0.48817725]
ouput values: [0.48733719005293535]

Testing Information
------------------------
input values: [0.52]
expected values: [0.49688014]
ouput values: [0.4977889413605415]

Testing Information
------------------------
input values: [0.55]
expected values: [0.52268723]
ouput values: [0.5287581361788347]

Testing Information
------------------------
input values: [0.57]
expected values: [0.53963205]
ouput values: [0.5489708912750744]

Testing Information
------------------------
input values: [0.58]
expected values: [0.54802394]
ouput values: [0.5588874234714049]

Testing Information
------------------------
input values: [0.59]
expected values: [0.55636102]
ouput values: [0.5686861510429841]

Testing Information
------------------------
input values: [0.6]
expected values: [0.56464247]
ouput values: [0.5783604738363185]

Testing Information
------------------------
input values: [0.62]
expected values: [0.58103516]
ouput values: [0.5974230510821419]

Testing Information
------------------------
input values: [0.63]
expected values: [0.58914476]
ouput values: [0.6066895690625438]

Testing Information
------------------------
input values: [0.64]
expected values: [0.59719544]
ouput values: [0.6158110653824914]

Testing Information
------------------------
input values: [0.65]
expected values: [0.60518641]
ouput values: [0.6247838157496038]

Testing Information
------------------------
input values: [0.67]
expected values: [0.62098599]
ouput values: [0.6424182372240623]

Testing Information
------------------------
input values: [0.68]
expected values: [0.62879302]
ouput values: [0.6509257954318148]

Testing Information
------------------------
input values: [0.7]
expected values: [0.64421769]
ouput values: [0.6676219927421492]

Testing Information
------------------------
input values: [0.71]
expected values: [0.65183377]
ouput values: [0.6756476232502113]

Testing Information
------------------------
input values: [0.72]
expected values: [0.65938467]
ouput values: [0.6835116026068709]

Testing Information
------------------------
input values: [0.73]
expected values: [0.66686964]
ouput values: [0.6912136948499397]

Testing Information
------------------------
input values: [0.74]
expected values: [0.67428791]
ouput values: [0.6987539775430707]

Testing Information
------------------------
input values: [0.75]
expected values: [0.68163876]
ouput values: [0.7061328162391785]

Testing Information
------------------------
input values: [0.76]
expected values: [0.68892145]
ouput values: [0.7133508399293698]

Testing Information
------------------------
input values: [0.77]
expected values: [0.69613524]
ouput values: [0.7204089176021654]

Testing Information
------------------------
input values: [0.78]
expected values: [0.70327942]
ouput values: [0.7273081360013481]

Testing Information
------------------------
input values: [0.79]
expected values: [0.71035327]
ouput values: [0.7340497786386524]

Testing Information
------------------------
input values: [0.81]
expected values: [0.72428717]
ouput values: [0.7471979016333972]

Testing Information
------------------------
input values: [0.84]
expected values: [0.74464312]
ouput values: [0.7658063505995734]

Testing Information
------------------------
input values: [0.85]
expected values: [0.75128041]
ouput values: [0.7716315173473274]

Testing Information
------------------------
input values: [0.86]
expected values: [0.75784256]
ouput values: [0.777312144276879]

Testing Information
------------------------
input values: [0.88]
expected values: [0.77073888]
ouput values: [0.7883348104969635]

Testing Information
------------------------
input values: [0.89]
expected values: [0.77707175]
ouput values: [0.7935939282639691]

Testing Information
------------------------
input values: [0.91]
expected values: [0.78950374]
ouput values: [0.8037740934545008]

Testing Information
------------------------
input values: [0.92]
expected values: [0.79560162]
ouput values: [0.8086338568711255]

Testing Information
------------------------
input values: [0.93]
expected values: [0.80161994]
ouput values: [0.8133659388632769]

Testing Information
------------------------
input values: [0.94]
expected values: [0.8075581]
ouput values: [0.8179728636022694]

Testing Information
------------------------
input values: [0.95]
expected values: [0.8134155]
ouput values: [0.8224571656891125]

Testing Information
------------------------
input values: [0.96]
expected values: [0.81919157]
ouput values: [0.8268213851200468]

Testing Information
------------------------
input values: [0.98]
expected values: [0.83049737]
ouput values: [0.8352202181144253]

Testing Information
------------------------
input values: [0.99]
expected values: [0.83602598]
ouput values: [0.8392389825504911]

Testing Information
------------------------
input values: [1.]
expected values: [0.84147098]
ouput values: [0.843147799056202]

Testing Information
------------------------
input values: [1.01]
expected values: [0.84683184]
ouput values: [0.8469491730856592]

Testing Information
------------------------
input values: [1.02]
expected values: [0.85210802]
ouput values: [0.8506455931249076]

Testing Information
------------------------
input values: [1.03]
expected values: [0.85729899]
ouput values: [0.8542395279896008]

Testing Information
------------------------
input values: [1.04]
expected values: [0.86240423]
ouput values: [0.8577334243506279]

Testing Information
------------------------
input values: [1.05]
expected values: [0.86742323]
ouput values: [0.8611297044698907]

Testing Information
------------------------
input values: [1.06]
expected values: [0.87235548]
ouput values: [0.8644307641305862]

Testing Information
------------------------
input values: [1.07]
expected values: [0.8772005]
ouput values: [0.8676389707482456]

Testing Information
------------------------
input values: [1.08]
expected values: [0.88195781]
ouput values: [0.8707566616504444]

Testing Information
------------------------
input values: [1.09]
expected values: [0.88662691]
ouput values: [0.8737861425145352]

Testing Information
------------------------
input values: [1.11]
expected values: [0.89569869]
ouput values: [0.879560347976156]

Testing Information
------------------------
input values: [1.12]
expected values: [0.90010044]
ouput values: [0.8823393366287987]

Testing Information
------------------------
input values: [1.13]
expected values: [0.90441219]
ouput values: [0.8850389814975786]

Testing Information
------------------------
input values: [1.14]
expected values: [0.9086335]
ouput values: [0.8876614116096887]

Testing Information
------------------------
input values: [1.15]
expected values: [0.91276394]
ouput values: [0.8902087178373094]

Testing Information
------------------------
input values: [1.16]
expected values: [0.91680311]
ouput values: [0.892682952224617]

Testing Information
------------------------
input values: [1.17]
expected values: [0.9207506]
ouput values: [0.8950861274186374]

Testing Information
------------------------
input values: [1.18]
expected values: [0.92460601]
ouput values: [0.8974202161991671]

Testing Information
------------------------
input values: [1.19]
expected values: [0.92836897]
ouput values: [0.8996871511031551]

Testing Information
------------------------
input values: [1.21]
expected values: [0.935616]
ouput values: [0.9039864015021698]

Testing Information
------------------------
input values: [1.22]
expected values: [0.93909936]
ouput values: [0.9060639486924796]

Testing Information
------------------------
input values: [1.23]
expected values: [0.9424888]
ouput values: [0.9080816441392277]

Testing Information
------------------------
input values: [1.24]
expected values: [0.945784]
ouput values: [0.9100412168739124]

Testing Information
------------------------
input values: [1.25]
expected values: [0.94898462]
ouput values: [0.9119443551784334]

Testing Information
------------------------
input values: [1.26]
expected values: [0.95209034]
ouput values: [0.9137927067486895]

Testing Information
------------------------
input values: [1.27]
expected values: [0.95510086]
ouput values: [0.9155878789188427]

Testing Information
------------------------
input values: [1.28]
expected values: [0.95801586]
ouput values: [0.9173314389422443]

Testing Information
------------------------
input values: [1.29]
expected values: [0.96083506]
ouput values: [0.9190249143250683]

Testing Information
------------------------
input values: [1.3]
expected values: [0.96355819]
ouput values: [0.9206697932087764]

Testing Information
------------------------
input values: [1.31]
expected values: [0.96618495]
ouput values: [0.9222675247976002]

Testing Information
------------------------
input values: [1.32]
expected values: [0.9687151]
ouput values: [0.9238195198273232]

Testing Information
------------------------
input values: [1.33]
expected values: [0.97114838]
ouput values: [0.9253271510717279]

Testing Information
------------------------
input values: [1.34]
expected values: [0.97348454]
ouput values: [0.9267917538831839]

Testing Information
------------------------
input values: [1.35]
expected values: [0.97572336]
ouput values: [0.9282146267639606]

Testing Information
------------------------
input values: [1.37]
expected values: [0.97990806]
ouput values: [0.9309056916490617]

Testing Information
------------------------
input values: [1.38]
expected values: [0.98185353]
ouput values: [0.9322115083016362]

Testing Information
------------------------
input values: [1.39]
expected values: [0.98370081]
ouput values: [0.9334804147684509]

Testing Information
------------------------
input values: [1.4]
expected values: [0.98544973]
ouput values: [0.934713535153175]

Testing Information
------------------------
input values: [1.41]
expected values: [0.9871001]
ouput values: [0.9359119606019195]

Testing Information
------------------------
input values: [1.42]
expected values: [0.98865176]
ouput values: [0.9370767500042287]

Testing Information
------------------------
input values: [1.43]
expected values: [0.99010456]
ouput values: [0.9382089307022033]

Testing Information
------------------------
input values: [1.44]
expected values: [0.99145835]
ouput values: [0.9393094992055608]

Testing Information
------------------------
input values: [1.47]
expected values: [0.99492435]
ouput values: [0.9423791257482385]

Testing Information
------------------------
input values: [1.49]
expected values: [0.99673775]
ouput values: [0.9442969734767322]

Testing Information
------------------------
input values: [1.5]
expected values: [0.99749499]
ouput values: [0.9452285456954631]

Testing Information
------------------------
input values: [1.51]
expected values: [0.99815247]
ouput values: [0.946134642385166]

Testing Information
------------------------
input values: [1.52]
expected values: [0.99871014]
ouput values: [0.9470160401889536]

Testing Information
------------------------
input values: [1.53]
expected values: [0.99916795]
ouput values: [0.947873491305078]

Testing Information
------------------------
input values: [1.54]
expected values: [0.99952583]
ouput values: [0.9487077241700591]

Testing Information
------------------------
input values: [1.55]
expected values: [0.99978376]
ouput values: [0.9495194441322607]

Testing Information
------------------------
input values: [1.56]
expected values: [0.99994172]
ouput values: [0.9503093341151744]

Testing Information
------------------------
input values: [1.57]
expected values: [0.99999968]
ouput values: [0.9510780552697552]

Final root-mean-square error: 0.03152512359669679


Process finished with exit code 0

"""