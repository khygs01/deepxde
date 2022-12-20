"""Boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "DirichletBC_xy",
    "NeumannBC",
    "OperatorBC",
    "PeriodicBC",
    "PointSetBC",
    "PointSetOperatorBC",
    "RobinBC",
]

import numbers
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from .. import backend as bkd
from .. import config
from .. import data
from .. import gradients as grad
from .. import utils
from ..backend import backend_name


class BC(ABC):
    """Boundary condition base class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

        self.boundary_normal = npfunc_range_autocache(
            utils.return_tensor(self.geom.boundary_normal)
        )

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end, None)
        return bkd.sum(dydx * n, 1, keepdims=True)

    @abstractmethod
    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        """Returns the loss."""
        # aux_var is used in PI-DeepONet, where aux_var is the input function evaluated
        # at x.


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class DirichletBC_xy(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, xy):
        values = self.func(X, beg, end, xy)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return xy[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(utils.return_tensor(func))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return self.normal_derivative(X, inputs, outputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component : self.component + 1]
            yright = outputs[mid:end, self.component : self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return yleft - yright


class SharedBdryBC(BC):
    def __init__(self, geom, on_boundary, component=None):
        # geom should be an instance of ``CSGIntersection``
        super().__init__(geom, on_boundary, component)

    @abstractmethod
    def error(self, X, xieta0, xieta1, xy0, xy1, uvp0, uvp1, beg, end):
        """Returns the loss."""


class SharedBdryXYBC(SharedBdryBC):
    def __init__(self, geom, on_boundary, component):
        super().__init__(geom, on_boundary, component)

    def error(self, X, xieta0, xieta1, xy0, xy1, uvp0, uvp1, beg, end):
        return (
            xy0[beg:end, self.component : self.component + 1]
            - xy1[beg:end, self.component : self.component + 1]
        )


class SharedBdryUVPBC(SharedBdryBC):
    def __init__(self, geom, on_boundary, component):
        super().__init__(geom, on_boundary, component)

    def error(self, X, xieta0, xieta1, xy0, xy1, uvp0, uvp1, beg, end):
        return (
            uvp0[beg:end, self.component : self.component + 1]
            - uvp1[beg:end, self.component : self.component + 1]
        )


class SharedBdryResidualBC(SharedBdryBC):
    def __init__(self, geom, on_boundary, func):
        super().__init__(geom, on_boundary)
        self.func = func  # pde

    def error(self, X, xieta0, xieta1, xy0, xy1, uvp0, uvp1, beg, end):
        return (
            self.func(xieta0, uvp0, xy0)[beg:end]
            - self.func(xieta1, uvp1, xy1)[beg:end]
        )


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geom: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    Warning:
        If you use `X` in `func`, then do not set ``num_test`` when you define
        ``dde.data.PDE`` or ``dde.data.TimePDE``, otherwise DeepXDE would throw an
        error. In this case, the training points will be used for testing, and this will
        not affect the network training and training loss. This is a bug of DeepXDE,
        which cannot be fixed in an easy way for all backends.
    """

    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, X)[beg:end]


class OperatorBC_uvpxy(BC):
    """General operator boundary conditions: func(xieta, uvp, xy) = 0."""

    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, None)
        self.func = func

    def error(self, X, xieta, uvp, beg, end, xy):
        return self.func(xieta, uvp, xy)[beg:end]


class FlowRateBC:
    """
    Assure flow rate values along specified lines. To be used with `FlowrateBdry` data class.
    Supports dim == 2 only.
    """

    def __init__(self, Q, N_quad, quad_method="legendre"):
        self.Q = Q  # specified flow rate
        self.N = N_quad  # number of quadrature points
        # quadrature method (legendre: Gauss-Legendre, lobatto: Gauss-Lobatto)
        self.quad = {
            "legendre": self.GaussLegendreQuadrature,
            "lobatto": self.GaussLobattoQuadrature,
        }[quad_method]
        if backend_name not in ["tensorflow.compat.v1", "tensorflow"]:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, and tensorflow supported"
            )

    def GaussLegendreQuadrature(self):
        from scipy.special import roots_legendre

        X, W = roots_legendre(self.N)
        return X, W

    def GaussLobattoQuadrature(self):
        from scipy.special import roots_jacobi, legendre

        N = self.N
        if N <= 1:
            raise ValueError("There must be more than 1 quadrature points!")
        W = []
        X, _ = roots_jacobi(N - 2, 1, 1)
        wl = wr = 2 / (N * (N - 1))
        w = 2 / (N * (N - 1) * legendre(N - 1)(X) ** 2)
        X = np.insert(X, [0, len(X)], [-1, 1])
        W = np.insert(w, [0, len(w)], [wl, wr])
        return X, W

    def error(self, xei, xef, net):
        from ..backend import tf

        xyi = net._build_net(
            xei,
            net.layers_xy,
            net._input_transform_xy,
            net._output_transform_xy,
            skip=True,
        )  # shape = (M, 2)
        xyf = net._build_net(
            xef,
            net.layers_xy,
            net._input_transform_xy,
            net._output_transform_xy,
            skip=True,
        )  # shape = (M, 2)
        xi, yi = xyi[:, 0:1], xyi[:, 1:]
        xf, yf = xyf[:, 0:1], xyf[:, 1:]
        # dxy = xyf - xyi  # shape = (M, 2)
        # dx, dy = dxy[:, 0], dxy[:, 1]  # shape = (M,)
        dx = xf - xi  # shape = (M, 1)
        dy = yf - yi
        dS = tf.sqrt(dx**2 + dy**2)  # shape = (M,)
        nx, ny = dy / dS, -dx / dS  # shape = (M,)
        pts, W = self.quad()  # shape = (N,)
        # xy = xyi + dxy * (pts + 1) / 2  # shape = (M, 2, N), from (-1, 1) to (xyi, xyf)
        x = xi + dx * (pts + 1) / 2  # shape = (M, N), from (-1, 1) to (xi, xf)
        y = yi + dy * (pts + 1) / 2
        xy = tf.stack((x, y), axis=-1)  # shape = (M, N, 2)
        # xy = tf.transpose(xy, [0, 2, 1])  # shape = (M, N, 2)
        uvp = net._build_net(
            xy, net.layers_uvp, net._input_transform_uvp, net._output_transform_uvp
        )  # shape = (M, N, 3)
        uv_n = nx * uvp[..., 0] + ny * uvp[..., 1]  # shape = (M, N)
        J = dS / 2  # shape = (M,)
        q_pred = J * tf.matmul(
            uv_n, W[:, np.newaxis]
        )  # (M, 1) * ((M, N) x (N, 1)) = (M, 1)
        return q_pred - self.Q


class PointSetBC:
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(
        self, points, values, component=0, batch_size=None, shuffle=True
    ):
        self.points = np.array(points, dtype=config.real(np))
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component
        if isinstance(component, list) and backend_name != "pytorch":
            # TODO: Add support for multiple components in other backends
            raise RuntimeError(
                "multiple components only implemented for pytorch backend"
            )
        self.batch_size = batch_size

        if batch_size is not None:  # batch iterator and state
            if backend_name != "pytorch":
                raise RuntimeError(
                    "batch_size only implemented for pytorch backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(
                len(self), shuffle=shuffle
            )
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if self.batch_size is not None:
            if isinstance(self.component, numbers.Number):
                return (
                    outputs[beg:end, self.component : self.component + 1]
                    - self.values[self.batch_indices]
                )
            return (
                outputs[beg:end, self.component]
                - self.values[self.batch_indices]
            )
        if isinstance(self.component, numbers.Number):
            return (
                outputs[beg:end, self.component : self.component + 1]
                - self.values
            )
        # When a concat is provided, the following code works 'fast' in paddle cpu,
        # and slow in both tensorflow backends, jax untested.
        # tf.gather can be used instead of for loop but is also slow
        # if len(self.component) > 1:
        #    calculated_error = outputs[beg:end, self.component[0]] - self.values[:,0]
        #    for i in range(1,len(self.component)):
        #        tmp = outputs[beg:end, self.component[i]] - self.values[:,i]
        #        calculated_error = bkd.lib.concat([calculated_error,tmp],axis=0)
        # else:
        #    calculated_error = outputs[beg:end, self.component[0]] - self.values
        # return calculated_error
        return outputs[beg:end, self.component] - self.values


class PointSetOperatorBC:
    """General operator boundary conditions for a set of points.
    
    Compare the function output, func, (that associates with `points`) 
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are 
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of 
            `inputs`. `inputs` and `outputs` are the network input and output 
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(self, points, values, func):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError("PointSetOperatorBC should output 1D values")
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.func = func

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, X)[beg:end] - self.values


def npfunc_range_autocache(func):
    """Call a NumPy function on a range of the input ndarray.

    If the backend is pytorch, the results are cached based on the id of X.
    """
    # For some BCs, we need to call self.func(X[beg:end]) in BC.error(). For backend
    # tensorflow.compat.v1/tensorflow, self.func() is only called once in graph mode,
    # but for backend pytorch, it will be recomputed in each iteration. To reduce the
    # computation, one solution is that we cache the results by using @functools.cache
    # (https://docs.python.org/3/library/functools.html). However, numpy.ndarray is
    # unhashable, so we need to implement a hash function and a cache function for
    # numpy.ndarray. Here are some possible implementations of the hash function for
    # numpy.ndarray:
    # - xxhash.xxh64(ndarray).digest(): Fast
    # - hash(ndarray.tobytes()): Slow
    # - hash(pickle.dumps(ndarray)): Slower
    # - hashlib.md5(ndarray).digest(): Slowest
    # References:
    # - https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array/16592241#16592241
    # - https://stackoverflow.com/questions/39674863/python-alternative-for-using-numpy-array-as-key-in-dictionary/47922199
    # Then we can implement a cache function or use memoization
    # (https://github.com/lonelyenvoy/python-memoization), which supports custom cache
    # key. However, IC/BC is only for dde.data.PDE, where the ndarray is fixed. So we
    # can simply use id of X as the key, as what we do for gradients.

    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end, _):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_nocache_auxiliary(X, beg, end, aux_var):
        return func(X[beg:end], aux_var[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end, _):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]

    @wraps(func)
    def wrapper_cache_auxiliary(X, beg, end, aux_var):
        # Even if X is the same one, aux_var could be different
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end], aux_var[beg:end])
        return cache[key]

    if backend_name in ["tensorflow.compat.v1", "tensorflow", "jax"]:
        if utils.get_num_args(func) == 1:
            return wrapper_nocache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
    if backend_name in ["pytorch", "paddle"]:
        if utils.get_num_args(func) == 1:
            return wrapper_cache
        if utils.get_num_args(func) == 2:
            return wrapper_nocache_auxiliary
