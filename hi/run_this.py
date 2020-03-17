import os
import numpy
import random
from main import *
import warnings
warnings.filterwarnings("ignore")

folder = "test"

def load_other_array(name_of_file):
    with open(folder + '/' + name_of_file) as f:
        array = []
        for line in f:
            sub_array = [float(x) for x in line.replace('\n', '').split('\t')][1:7]
            array.append(sub_array)
    return array


for filename in os.listdir(folder):
    print('file name:', filename)
    print('')
    # LOAD IN THOSE GESTURES MAN
    other = load_other_array(filename)
    # print(other)
    observation_vector = []
    for i in other:
        observation_vector.append(k_nn(loaded_clusters, i))
    which_gesture(observation_vector)
    print('')
    