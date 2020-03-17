import numpy

from helper import *
from baum_welch_helper import *
import csv
import warnings
warnings.filterwarnings("ignore")

def alpha_beta_calc(pi, B, A, observation_vector, alpha, beta, c):

    # init T and N
    T = len(observation_vector)
    N = len(A)
    
    # populate the c initialization
    for n in range(N):
        alpha[n][0] = pi[n] * B[n][observation_vector[0]]
        c[0] += alpha[n][0]  
    c[0] = 1/c[0]

    # alpha initialization
    for n in range(N):
        alpha[n][0] = c[0]*alpha[n][0] + 0.00000000001
        
    # alpha induction step
    for t in range(1,T):
        c[t] = 0
        for i in range(N):
            alpha[i][t] = 0
            for j in range(N):
                alpha[i][t] += alpha[j][t-1]*A[j][i] 
            alpha[i][t] = alpha[i][t]*B[i][observation_vector[t]] + 0.00000000001
            c[t] += alpha[i][t]
        c[t] = 1/c[t] 
        for i in range(N): 
            alpha[i,t] = c[t]*alpha[i][t] 
    
    # beta init should be 1's in T-1 column
    # scaled = 1 * scaling factor
    for i in range (N):
        beta[i][T-1] = 1 * c[T-1]
        
    # populate beta array
    for k in range(1,T):
        t = T-(k+1)
        for n in range (N):
            beta[n][t] = 0
            for j in range(N):
                beta[n][t] += A[n][j] * B[j][observation_vector[t+1]] * beta[j][t+1] 
            beta[n][t] = c[t] * beta[n][t]


    for i in range(len(c)):
        c[i] = c[i]+.000000001

    log_likelihood = -numpy.sum(numpy.log(c))
    # termination gives us probability of observation given lambda
    P_O_given_lambda = numpy.sum(alpha[:,T-1])

    return alpha, beta, c, P_O_given_lambda, log_likelihood
    

def xi_gamma_calc(alpha, beta, A, B, observation_vector, xi,  gamma):
    # init T and N
    T = len(observation_vector)
    N = len(A)

    # populate gamma and xi
    for t in range(T-1):
        for i in range(N):
            gamma[i][t] = 0
            for j in range(N):
                xi[i][t][j] = alpha[i][t] * A[i][j] * B[j][observation_vector[t+1]] * beta[j][t+1]
                gamma[i][t] += xi[i][t][j]
                
    # the last column of gamma is alpha's
    gamma[:,T-1] = alpha[:,T-1]

    return gamma, xi

def update(xi, pi, alpha, beta, gamma, A, B, observation_vector):

    # initialize N,M,T
    N = len(A)
    M = len(B[0])
    T = len(observation_vector)

    # pi initializaton
    for n in range(N):
        pi[n] = gamma[n][0]

    # B update using gamma
    for n in range(N):
        denom = 0 
        for t in range(T):
            denom += gamma[n][t]
        for j in range(M):
            numer = 0
            for t in range(T):
                if (observation_vector[t] == j):
                    numer += gamma[n][t]
            B[n][j] = numer/denom
     
    # A update using gamma and
    for n in range(N):
        denom = 0
        for t in range(T-1):
            denom += gamma[n][t]
        for j in range(N):
            numer = 0
            for t in range(T-1):
                numer += xi[n][t][j]
            A[n][j] = numer/denom

    return pi, A, B

def printer(l, A, B, alpha, beta, xi, gamma, pi, c):

    print('iteration')
    print(l)

    # print('c')
    # print(c)

    print('alpha')
    print(alpha)

    # print('beta')
    # print(beta)

    # print('xi')
    # print(xi)

    # print('gamma')
    # print(gamma)

    # print('updated pi')
    # print(pi)

    # print('updated A')
    # print(A)

    # print('updated B')
    # print(B)



    return