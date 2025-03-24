import os
import pandas as pd 
import numpy as np

def get_time_domain():
    return np.arange(0,0.01,0.0001)

def read_file(filename):
    data = pd.read_csv(filename,sep='\t')
    data = cut_to_start(data)
    return np.array(data)[0]

def cut_to_start(data):
    start = data.where(data["Average [m/s]"]>0.001)["Time [s]"].idxmin()
    data =  data.iloc[start:start+100]
    data = pd.DataFrame(data['Average [m/s]']).transpose()
    return data

def get_n_data_samples(n):
    y = []
    for i in range(n):
        if i<9:
            y.append(read_file(f'./Data/P0{i+1}A_Center01A_VeloY.xls'))
        else:
            y.append(read_file(f'./Data/P{i+1}A_Center01A_VeloY.xls'))
    return y

def scale_data(y):
    for i in range(len(y)):
        y[i] = y[i]*100