''' 

proj 2
lois lee

'''

from helper import *
import csv

beat3, beat4, inf, circle, eight, wave = load()
everything = beat3 + beat4 + inf + circle + eight + wave

out = []
for i in everything:
    for j in i:
        out.append(j)
# out=out[0:10]
cluster_centers, categorized = k_means(out, 50)

print(cluster_centers)
print(categorized)

with open('cluster_centers.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(cluster_centers)

with open('categorized.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(categorized)

print(categorized)

# print(beat3)



