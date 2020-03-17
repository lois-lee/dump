from helper import *
from baum_welch import *
from baum_welch_helper import *
import csv
import warnings
warnings.filterwarnings("ignore")

A_array  = []
pi_array = []
B_array  = []

# LOAD IN THE FUCKING PI AND A AND B
with open('pi.csv') as f:
    array = []
    for line in f:
        sub_array = [float(x) for x in line.replace('\n', '').split(',')]
        pi_array.append(sub_array)

with open('A.csv') as f:
    large = []
    array = []
    for line in f:
        array.append(line)
    intermediate = ' '.join(array)
    intermediate = intermediate.replace('\n', '').replace('","',',') .replace(' ',',').replace(',,',',').split('","')
    for each in intermediate:
        middle = []
        each = each.split('],[')
        for i in each:
            mini = [float(x) for x in i.replace('[','').replace(']','').replace('"','').split(',')]
            middle.append(mini)
        large.append(middle)
    A_array = large 

with open('B.csv') as f:
    large = []
    array = []
    for line in f:
        array.append(line)
    intermediate = ' '.join(array)
    intermediate = intermediate.replace('\n', '').replace('","',',') .replace(' ',',').replace(',,',',').split('","')
    large = []
    for each in intermediate:
        each = each.split('],[')
        middle = []
        for i in each:
            call = []
            mini = [(x) for x in i.replace('[','').replace(']','').replace('"','').split(',')]
            for x in mini:
                if x!='':
                    call.append(float(x))
            middle.append(call)
        large.append(middle)
    B_array = large 

folder = "kms"
gestures = []
for filename in os.listdir(folder):
    print(filename)
    each = []
    with open(folder + '/' + filename) as f:
        array = []
        for line in f:
            sub_array = [float(x) for x in line.replace('\n', '').split('\t')][1:7]
            array.append(sub_array)
        each.append(array)
    gestures.append(each)



# LOAD IN THE FUCKING CLUSTERS
loaded_clusters = loading_for_cluster_centers()


# LOAD IN THOSE GESTURES MAN
quantized = []
for each_gesture in gestures:
    gesture_quantized = []
    for i in range(len(each_gesture[0])):
        gesture_quantized.append(k_nn(loaded_clusters, each_gesture[0][i]))
    quantized.append(gesture_quantized)



def which_gesture(observation_vector):
    T = len(observation_vector)
    N = 10

    # initialize params
    alpha = numpy.zeros((N, T))
    beta = numpy.zeros((N, T))
    c = numpy.zeros(T)

    # which one is it
    a, b, c_wave, likelihood_wave, log_likelihood_wave   = alpha_beta_calc(pi_array[0], B_array[0], A_array[0], observation_vector, alpha, beta, c)
    a, b, c_beat3, likelihood_beat3, log_likelihood_beat3  = alpha_beta_calc(pi_array[1], B_array[1], A_array[1], observation_vector, alpha, beta, c)
    a, b, c_beat4, likelihood_beat4, log_likelihood_beat4  = alpha_beta_calc(pi_array[2], B_array[2], A_array[2], observation_vector, alpha, beta, c)
    a, b, c_inf, likelihood_inf, log_likelihood_inf    = alpha_beta_calc(pi_array[3], B_array[3], A_array[3], observation_vector, alpha, beta, c)
    a, b, c_circle, likelihood_circle, log_likelihood_circle = alpha_beta_calc(pi_array[4], B_array[4], A_array[4], observation_vector, alpha, beta, c)
    a, b, c_eight, likelihood_eight, log_likelihood_eight  = alpha_beta_calc(pi_array[5], B_array[5], A_array[5], observation_vector, alpha, beta, c)



    likelihoods = [log_likelihood_wave,log_likelihood_beat3,log_likelihood_beat4,log_likelihood_inf,log_likelihood_circle,log_likelihood_eight]
    # print(likelihoods)
    
    output_gest = ['wave', 'beat3', 'beat4', 'inf', 'circle', 'eight']
    # print(output_gest)

    for i in range(len(likelihoods)):
        likelihoods[i] = (abs(likelihoods[i]))

    mini = 10000000000000000000000
    min_index = -1
    

    godfml = [x for x in likelihoods]
    for i in range(len(likelihoods)):
        if likelihoods[i] <= mini:
            mini = likelihoods[i]
            min_index = i
    godfml[min_index] = 10000000000000000000000
    mini = 10000000000000000000000
    second_index = -1
    third_index = -1
    for i in range(len(likelihoods)):
        if godfml[i] <= mini:
            mini = godfml[i]
            second_index = i
    godfml[second_index] = 10000000000000000000000
    mini = 10000000000000000000000
    third_index = -1
    for i in range(len(likelihoods)):
        if godfml[i] <= mini:
            mini = godfml[i]
            third_index = i
    godfml[third_index] = 10000000000000000000000

            # third_index = second_index
            # second_index = min_index
            # min_index = i


    print('first most likely')
    print(output_gest[min_index],': log likelihood = ', -likelihoods[min_index])
    print('second most likely')
    print(output_gest[second_index],': log likelihood = ', -likelihoods[second_index])
    print('third most likely')
    print(output_gest[third_index],': log likelihood = ', -likelihoods[third_index])
    print('_________________________________')
    # print(likelihoods)
    return output_gest[min_index]
        

which_gesture(quantized[0])
which_gesture(quantized[1])
which_gesture(quantized[2])
which_gesture(quantized[3])
which_gesture(quantized[4])
which_gesture(quantized[5])
# print('_________________________________')


