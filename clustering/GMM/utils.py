import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_ellipses(mu, sigma, ax, color='red'):
    K = mu.shape[0]
    for k in range(K):
        cov = sigma[k][:2,:2]
        v, w = np.linalg.eigh(cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mu[k, :2], v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')