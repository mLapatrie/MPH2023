
import numpy as np
import pandas as pd
from random import random

common_labels = [1, 5, 6, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 31, 32, 33, 34, 35, 37, 38, 39, 40, 48, 49, 50, 51, 53, 54, 62, 63, 64, 65, 67, 71, 73, 74, 76, 80, 81, 87, 88, 89, 90, 91, 92, 93, 96, 98, 99, 106, 107, 109, 110, 112, 113, 114, 115, 123, 124, 125, 126, 128, 129, 137, 138, 139, 140, 142, 146, 148, 149]
rois_labels = [17, 18, 33, 34, 49, 87, 92, 107, 109, 110, 124]


def generate_one():
    data = np.zeros(len(common_labels))

    for i in range(len(common_labels)):
        data[i] = random()*0.5 - 0.25

    for i in rois_labels:
        ind = common_labels.index(i)
        data[ind] = 0.5 + random()*0.4 - 0.1
    
    return data
    
data_fin = []
for i in range(505):
    data_fin.append(generate_one())
    
np.savetxt('big_data.csv', data_fin, delimiter=',')
