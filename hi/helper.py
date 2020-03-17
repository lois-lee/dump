''' 

proj 2
lois lee

'''
import os
import numpy
import random


def load_array(name_of_file):
    folder = "ECE5242Proj2-train"
    with open(folder + '/' + name_of_file) as f:
        array = []
        for line in f:
            sub_array = [float(x) for x in line.replace('\n', '').split('\t')][1:7]
            array.append(sub_array)
    return array

def loading_for_cluster_centers():
    name_of_file = "cluster_centers.csv"
    with open(name_of_file) as f:
        total = []
        for line in f:
            total.append(line)
        lined = (''.join(total))
        lined = lined.replace('"', '').replace('\n', '').replace('[', '').replace(']', '').split(',')
        
        output = []
        for i in lined:
            splitttttt = i.split(' ')
            sub = []
            for j in splitttttt:
                if j != '':
                    sub.append(float(j))
            output.append(sub)
    print('cluster centers have been imported....')
    return output

# print(loading_for_cluster_centers())

def loading_for_categorized():
    name_of_file = "categorized.csv"
    with open(name_of_file) as f:
        for line in f:
            output = [int(x) for x in (line.split(','))]
    print('the quantized data has been imported....')
    return output

# print(loading_for_categorized())


def load():
    folder = "ECE5242Proj2-train"
    beat3, beat4, inf, circle, eight, wave = [], [], [], [], [], []

    for filename in os.listdir(folder):
        if 'beat3' in filename:
            beat3.append(load_array(filename))
        elif 'beat4' in filename:
            beat4.append(load_array(filename))
        elif 'inf' in filename:
            inf.append(load_array(filename))
        elif 'circle' in filename:
            circle.append(load_array(filename))
        elif 'eight' in filename:
            eight.append(load_array(filename))
        elif 'wave' in filename:
            wave.append(load_array(filename))
        else:
            print('not a valid training image')

    print('arrays have been loaded.......')
    return beat3, beat4, inf, circle, eight, wave

# k_nn
# takes in a list of centers (which are represented as n-dimensional vectors) and a new point
# 
def k_nn(centers, point):
    closest_distance, closest_center = 100000000000000, []
    for i in range(len(centers)):
        if abs(numpy.linalg.norm(numpy.array(centers[i])-numpy.array(point))) < closest_distance:
            closest_distance, closest_center = abs(numpy.linalg.norm(numpy.array(centers[i])-numpy.array(point))), i
    return closest_center

def init_clusters(points, k):
    cluster_centers = []
    while (len(cluster_centers) != k):
        cen = random.choice(points)
        if cen not in cluster_centers:
            cluster_centers.append(cen)
    return cluster_centers

# print(init_clusters([[1,3],[3,3],[6,6],[7,8]], 3))

def categorize(cluster_centers, points):
    categorized = []
    for i in points:
        categorized.append(k_nn(cluster_centers, i))
    return categorized

# print(categorize([[1,1],[5,6]],[[2,1],[5,6]]))

def k_means(points, k):
    cluster_centers, iterations = init_clusters(points, k), 0
    while (iterations < 50):
        categorized, new_centroids = categorize(cluster_centers, points), []


        for m in range(k):
            summed, num = numpy.zeros(len(points[0])), 0.0
            for i in range(len(categorized)):
                if categorized[i] == m:
                    num += 1.0
                    summed += points[i]
            new_centroids.append(summed/float(num))

        cluster_centers = new_centroids
        iterations  += 1
        print('.')
    # print(categorized)
    return cluster_centers, categorized

# print(k_means([[1,3,5,6,7,6],[3,3,7,4,5,3]], 2))

def something_dumb(arr):
    output = []
    for i in arr:
        for j in i:
            output.append(j)
    return output