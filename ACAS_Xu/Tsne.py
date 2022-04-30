import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pickle_path = './results/tsne_crash.pkl'
with open(pickle_path, 'rb') as handle:
    X1 = pickle.load(handle)
pickle_path = './results/tsne_nocrash.pkl'
with open(pickle_path, 'rb') as handle:
    X2 = pickle.load(handle)
pickle_path = './results/tsne_random.pkl'
with open(pickle_path, 'rb') as handle:
    X3 = pickle.load(handle)

X1_new = []
X2_new = []
X3_new = []
for i in range(len(X2)):
    for j in range(len(X2[i])):
        X2_new.append(np.hstack((X2[i][j][0], X2[i][j][1])))
for i in range(len(X3)):
    for j in range(len(X3[i])):
        X3_new.append(np.hstack((X3[i][j][0], X3[i][j][1])))
for i in range(len(X1)):
    X1_new.append(np.hstack((X1[i][0], X1[i][1])))
X1 = np.array(X1_new)
X2 = np.array(X2_new)
X3 = np.array(X3_new)
X1 = X1.reshape((X1.shape[0], X1.shape[2]))
X2 = X2.reshape((X2.shape[0], X2.shape[2]))
X3 = X3.reshape((X3.shape[0], X3.shape[2]))

X = np.vstack((X1, X2, X3))

Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])
Y3 = np.zeros(X3.shape[0])

Y = np.append(np.append(Y1, Y2), Y3)

# X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded = np.load('./results/TSNE_save.npy')


plt.scatter(X_embedded[:X1.shape[0], 0], X_embedded[:X1.shape[0], 1], c='red', label='Crashed', s=0.3)
plt.scatter(X_embedded[X1.shape[0]: X1.shape[0]+X2.shape[0], 0], X_embedded[X1.shape[0]:X1.shape[0]+X2.shape[0], 1], c='blue', label='Normal', s=0.3)
plt.scatter(X_embedded[X1.shape[0]+X2.shape[0]:, 0], X_embedded[X1.shape[0]+X2.shape[0]:, 1], c='green', label='Random', s=0.3)

plt.axes().xaxis.set_major_formatter(plt.NullFormatter())
plt.axes().yaxis.set_major_formatter(plt.NullFormatter())

plt.grid(True)
plt.savefig('TSNE-ACAS.pdf', transparent=True, bbox_inches='tight')