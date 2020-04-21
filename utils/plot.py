import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

FIGURES_DIR = 'figures/'


def plot_data(features, labels):
    x, y, z, p = features['income'], features['age'], features['loan'], labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=p)
    ax.set_xlabel('Income', fontsize=14)
    ax.set_ylabel('Age', fontsize=14)
    ax.set_zlabel('Loan', fontsize=14)
    fig.colorbar(img)
    plt.savefig(FIGURES_DIR + f'Figure_data' + '.png')
    plt.show()


def plot_cm(cm, meth: str = None):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.savefig(FIGURES_DIR + f'Figure_cm_{meth}_' + '.png')
    plt.show()


def plot_pca(features, labels, title, meth):
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.title(label=title)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap)
    plt.colorbar()
    plt.savefig(FIGURES_DIR + f'Figure_pca_{title}_{meth}_' + '.png')
    plt.show()
