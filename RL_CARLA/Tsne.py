import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X1 = np.load('./results/tsne_crash.pkl')
X2 = np.load('./results/tsne_normal.pkl')
X3 = np.load('./results/tsne_random.pkl')

X = np.vstack((X1, X2, X3))

print(X.shape)

Y1 = np.ones(X1.shape[0])
Y2 = np.zeros(X2.shape[0])
Y3 = np.zeros(X3.shape[0])

Y = np.append(np.append(Y1, Y2), Y3)

print(Y.shape)

# X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded = np.load('./results/TSNE_stored.npy')
print(X_embedded.shape)
# np.save('./TSNE_stored.npy', X_embedded)

plt.scatter(X_embedded[:X1.shape[0], 0], X_embedded[:X1.shape[0], 1], c='red', label='Crashed', s=0.3)
plt.scatter(X_embedded[X1.shape[0]: X1.shape[0]+X2.shape[0], 0], X_embedded[X1.shape[0]:X1.shape[0]+X2.shape[0], 1], c='blue', label='Normal', s=0.3)
plt.scatter(X_embedded[X1.shape[0]+X2.shape[0]:, 0], X_embedded[X1.shape[0]+X2.shape[0]:, 1], c='green', label='Random', s=0.3)

plt.axes().xaxis.set_major_formatter(plt.NullFormatter())
plt.axes().yaxis.set_major_formatter(plt.NullFormatter())

plt.grid(True)
plt.savefig('./results/TSNE_carla.pdf', transparent=True, bbox_inches='tight')