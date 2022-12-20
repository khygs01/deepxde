import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import alphashape
from shapely.geometry import Point


def concaveTriang(x, y):
    """
    Returns matplotlib.tri.Triangulation object given the x, y with shape (N,)
    Triangles outside of the concave hull of given points is masked out.
    """
    # create the concave hull (alpha=2.0) of the computed domain
    points = list(zip(x, y))
    alpha_shape = alphashape.alphashape(points, 2.0)

    triang = tri.Triangulation(x, y)
    # compute center of triangles
    x_c = x[triang.triangles].mean(axis=1)
    y_c = y[triang.triangles].mean(axis=1)
    # mask out triangles whose centers are outside of the concave hull
    mask = np.where(
        [alpha_shape.contains(Point(x_c_, y_c_)) for x_c_, y_c_ in zip(x_c, y_c)], 0, 1
    )
    triang.set_mask(mask)

    return triang


def _plot(fig, ax, x, y, z, title, cmap, N=100):
    ax.contour(x, y, z, N, cmap=cmap)
    ax.contourf(x, y, z, N, cmap=cmap)
    # ax.set_aspect("equal")
    _decorate_plot(fig, ax, title, z, cmap)


def _triplot(fig, ax, triang, z, title, cmap, N=100):
    ax.tricontour(triang, z, N, cmap=cmap)
    ax.tricontourf(triang, z, N, cmap=cmap)
    _decorate_plot(fig, ax, title, z, cmap)


def _decorate_plot(fig, ax, title, z, cmap):
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = mpl.colors.Normalize(np.nanmin(z), np.nanmax(z))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)


def _plotxymapping(xieta, xy, cmap, show=True):
    num_dom = len(xieta)
    fig, axes = plt.subplots(3, num_dom, figsize=(5 * num_dom, 10))
    i = 0
    if num_dom == 1:
        axes = np.array([axes], dtype=axes.dtype)
        axes = axes.T
    for (x_test_, y_test_), (x, y), ax in zip(xieta, xy, axes.T):
        _plot(fig, ax[0], x_test_, y_test_, x, "x_{}".format(i), cmap)
        _plot(fig, ax[1], x_test_, y_test_, y, "y_{}".format(i), cmap)
        N = int(np.sqrt(len(x_test_.flatten())))
        N = N // 20
        ax[2].scatter(x_test_[::N, ::N].flatten(), y_test_[::N, ::N].flatten())
        ax[2].scatter(x[::N, ::N].flatten(), y[::N, ::N].flatten())
        ax[2].set_title("xieta_xy_{}".format(i))
        i += 1
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plotResult(model, x_tests, y_tests, show=True):
    xy_test = []
    xieta = []
    for x_test, y_test in zip(x_tests, y_tests):
        x_test_, y_test_ = np.meshgrid(x_test, y_test)
        xy_test.append(
            np.concatenate((x_test_.reshape((-1, 1)), y_test_.reshape((-1, 1))), 1)
        )
        xieta.append((x_test_, y_test_))  # for plotting xieta vs. xy in each domain
    uvp, xy = model.predict(xy_test)
    uvp_concat = np.concatenate([uvp_ for uvp_ in uvp], axis=0)
    xy_concat = np.concatenate([xy_ for xy_ in xy], axis=0)

    u_, v_, p_, x_, y_ = (
        uvp_concat[:, 0],
        uvp_concat[:, 1],
        uvp_concat[:, 2],
        xy_concat[:, 0],
        xy_concat[:, 1],
    )
    xy = [
        (xy_[:, 0].reshape(x_test_.shape), xy_[:, 1].reshape(y_test_.shape))
        for xy_, (x_test_, y_test_) in zip(xy, xieta)
    ]  # for plotting xieta vs. xy in each domain

    triang = concaveTriang(x_, y_)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes
    cmap = mpl.cm.jet

    _triplot(fig, ax1, triang, u_, "u", cmap)
    _triplot(fig, ax2, triang, v_, "v", cmap)
    _triplot(fig, ax3, triang, p_, "p", cmap)

    fig.tight_layout()
    if show:
        plt.show()

    # plot mapping from (xi, eta) to (x, y)
    fig_map = _plotxymapping(xieta, xy, cmap, show)
    if not show:
        return fig, fig_map
