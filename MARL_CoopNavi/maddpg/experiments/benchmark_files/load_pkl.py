import numpy as np
import pickle

with open('./spread.pkl', 'rb') as f:
    result = pickle.load(f)
    for i in range(len(result)):
        print(result[i])
        break
    print(len(result))
