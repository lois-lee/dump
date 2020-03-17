import numpy

from helper import *
from baum_welch import *
from baum_welch_helper import *
import csv
import warnings
warnings.filterwarnings("ignore")


# LOAD & PROCESS THAT DATA
beat3, beat4, inf, circle, eight, wave = load()
beat3, beat4, inf, circle, eight, wave  = something_dumb(beat3), something_dumb(beat4), something_dumb(inf), something_dumb(circle), something_dumb(eight), something_dumb(wave) 
lengths = [int(len(beat3)), int(len(beat4)), int(len(inf)), int(len(circle)), int(len(eight)), int(len(wave))]
for i in range(len(lengths)):
    if (i != 0):
        lengths[i] = lengths[i] + lengths[i-1]

# LOAD the CLUSTERS and the CATEGORIZED data
loaded_clusters = loading_for_cluster_centers()
loaded_categorized = loading_for_categorized()
beat3, beat4, inf, circle, eight, wave = splitter(loaded_categorized, lengths)

def baum_welch(observation_vector):
    # parameters:
    num_clusters = 50
    num_hidden_states = 10

    T = len(observation_vector)
    N = num_hidden_states

    # initialize lambda
    pi, A, B = init_lambda(num_hidden_states, num_clusters)

    # initialize params
    alpha = numpy.zeros((N, T))
    beta = numpy.zeros((N, T))
    gamma = numpy.zeros((N,T))
    xi = numpy.zeros((N,T,N))
    c = numpy.zeros(T)

    # initialize log_likelihood
    log_likelihood = 100000
    l_max = 5

    # while (abs(log_likelihood) > 20000):
    for l in range(l_max):

        # forward-backward calculation
        alpha, beta, c, P_O_given_lambda, log_likelihood = alpha_beta_calc(pi, B, A, observation_vector, alpha, beta, c)
        # E-Step
        gamma, xi = xi_gamma_calc(alpha, beta, A, B, observation_vector, xi,  gamma)
        # M-Step
        updated_pi, updated_A, updated_B = update(xi, pi, alpha, beta, gamma, A, B, observation_vector)

        # log likelihood 
        log_likelihood = 0
        # for t in range(T):
        #     log_likelihood = log_likelihood + numpy.log(c[t])
        # log_likelihood = -1*log_likelihood
        log_likelihood = -numpy.sum(numpy.log(c))
        
        # print statements
        printer(l, A, B, alpha, beta, xi, gamma, pi, c)
        print('log_likelihood')
        print(log_likelihood)
        print('___________________________________________________________')

        pi, A, B = updated_pi, updated_A, updated_B

    print(pi, A, B)
    return pi, A, B

pi_wave, A_wave, B_wave  = baum_welch(wave)
pi_beat3, A_beat3, B_beat3  = baum_welch(beat3)
pi_beat4, A_beat4, B_beat4  = baum_welch(beat4)
pi_inf, A_inf, B_inf  = baum_welch(inf)
pi_circle, A_circle, B_circle = baum_welch(circle)
pi_eight, A_eight, B_eight  = baum_welch(eight)

with open('pi.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(pi_wave)
    writer.writerow(pi_beat3)
    writer.writerow(pi_beat4)
    writer.writerow(pi_inf)
    writer.writerow(pi_circle)
    writer.writerow(pi_eight)
    

with open('A.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(A_wave)
    writer.writerow(A_beat3)
    writer.writerow(A_beat4)
    writer.writerow(A_inf)
    writer.writerow(A_circle)
    writer.writerow(A_eight)


with open('B.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(B_wave)
    writer.writerow(B_beat3)
    writer.writerow(B_beat4)
    writer.writerow(B_inf)
    writer.writerow(B_circle)
    writer.writerow(B_eight)