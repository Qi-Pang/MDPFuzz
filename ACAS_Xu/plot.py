import numpy as np
import pickle

if __name__ == '__main__':
    plot_type = 'tsne'
    if plot_type == 'repair-reward':
        file1 = open('./results/repair_log.txt', 'r')
        Lines = file1.readlines()
        flag = False
        for line in Lines:
            if flag:
                print(line.split(', ')[2][14:-1])
                flag = False
            if line[0] == 'f':
                flag = True
    elif plot_type == 'tsne':
        pickle_path = './results/tsne_crash.pkl'
        with open(pickle_path, 'rb') as handle:
            result = pickle.load(handle)
        print(result[0])
