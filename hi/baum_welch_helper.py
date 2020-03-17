import numpy

from helper import *
import csv


# print('len')
# print(lengths)


def splitter(arr, lengths):
    beat3  = arr[0:lengths[0]]
    beat4  = arr[lengths[0]:lengths[1]]
    inf    = arr[lengths[1]:lengths[2]]
    circle = arr[lengths[2]:lengths[3]]
    eight  = arr[lengths[3]:lengths[4]]
    wave   = arr[lengths[4]:lengths[5]]
    return beat3, beat4, inf, circle, eight, wave


# initialize the lambda 
# N = number of hidden states
# M = number of k clusters
def init_lambda(N, M):
    pi, A, B =  numpy.zeros(N), numpy.zeros((N,N)), numpy.zeros((N,M))
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = 1/N
    for i in range(len(B)):
        for j in range(len(B[0])):
            B[i][j] = 1/M

    pi[0] = 1

    return pi, A, B

def distinct_number(arr):
    return len(set(arr))



# init_lambda(5,3)