import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

with open('./results/tsne_crash.pkl', 'rb') as handle:
    dataf = pickle.load(handle)
dataf = np.array(dataf)

X1 = dataf[7425: 7525]
X1 = np.vstack((X1, dataf[12930: 13030]))
X1 = np.vstack((X1, dataf[13555: 13655]))
X1 = np.vstack((X1, dataf[19286: 19386]))
X1 = np.vstack((X1, dataf[19853: 19953]))
X1 = np.vstack((X1, dataf[20634: 20734]))
X1 = np.vstack((X1, dataf[22593: 22693]))
X1 = np.vstack((X1, dataf[23011: 23111]))
X1 = np.vstack((X1, dataf[25595: 25695]))
X1 = np.vstack((X1, dataf[25986: 26086]))
X1 = np.vstack((X1, dataf[30236: 30336]))
X1 = np.vstack((X1, dataf[33409: 33509]))
X1 = np.vstack((X1, dataf[37306: 37406]))
X1 = np.vstack((X1, dataf[47005: 47105]))
X1 = np.vstack((X1, dataf[52757: 52857]))
X1 = np.vstack((X1, dataf[52935: 53035]))
X1 = np.vstack((X1, dataf[53326: 53426]))
X1 = np.vstack((X1, dataf[54604: 54704]))
X1 = np.vstack((X1, dataf[55057: 55157]))
X1 = np.vstack((X1, dataf[55778: 55878]))
X1 = np.vstack((X1, dataf[61371: 61471]))
X1 = np.vstack((X1, dataf[61761: 61861]))
X1 = np.vstack((X1, dataf[63913: 64013]))
X1 = np.vstack((X1, dataf[66091: 66191]))
X1 = np.vstack((X1, dataf[67950: 68050]))
X1 = np.vstack((X1, dataf[68389: 68489]))
X1 = np.vstack((X1, dataf[69960: 70060]))
X1 = np.vstack((X1, dataf[70761: 70861]))
X1 = np.vstack((X1, dataf[88555: 88655]))
X1 = np.vstack((X1, dataf[93211: 93311]))
X2 = dataf[:3000]
X3 = dataf[3700:6700]
X = np.vstack((X1, X2, X3))
Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])
Y3 = np.zeros(X3.shape[0])
Y = np.append(np.append(Y1, Y2), Y3)

X_embedded = np.load('./results/TSNE_stored.npy')

print(X_embedded.shape)
plt.scatter(X_embedded[:X1.shape[0], 0], X_embedded[:X1.shape[0], 1], c='red', label='Crashed', s=0.3)
plt.scatter(X_embedded[X1.shape[0]: X1.shape[0]+X2.shape[0], 0], X_embedded[X1.shape[0]:X1.shape[0]+X2.shape[0], 1], c='blue', label='Normal', s=0.3)
plt.scatter(X_embedded[X1.shape[0]+X2.shape[0]:, 0], X_embedded[X1.shape[0]+X2.shape[0]:, 1], c='green', label='Random', s=0.3)

plt.axes().xaxis.set_major_formatter(plt.NullFormatter())
plt.axes().yaxis.set_major_formatter(plt.NullFormatter())

plt.grid(True)
plt.savefig('./results/TSNE-RL-Game.pdf', transparent=True, bbox_inches='tight')
