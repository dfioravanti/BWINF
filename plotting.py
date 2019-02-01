import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_dataset(xs, ys, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    red_dots = xs[ys == 1]
    blue_dots = xs[ys == 0]
    
    ax.scatter(red_dots[:, 0], red_dots[:, 1], c='red', cmap=plt.cm.coolwarm, edgecolors='k', s=20, label='Class 1')
    ax.scatter(blue_dots[:, 0], blue_dots[:, 1], c='blue', cmap=plt.cm.coolwarm, edgecolors='k', s=20, label='Class 2')

    ax.legend()
    
def plot_classifier(xs, ys, clf):

    fig, ax = plt.subplots(1, 1)
    
    x0, x1 = xs[:, 0], xs[:, 1]
    xx, yy = make_meshgrid(x0, x1)    
    color = ['red' if l == 1 else 'blue' for l in ys]

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plot_dataset(xs, ys, ax)
