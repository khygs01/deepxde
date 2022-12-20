import numpy as np

from .data import Data
from .. import backend as bkd
from .. import config
from ..backend import backend_name
from ..utils import run_if_all_none


class FlowrateBdry(Data):
    """Assure flow rate q = q* along M lines"""

    def __init__(self, bc, net_idx, num_lines=0, vars=None, consts=None, var_xe="xi"):
        self.bc = bc
        self.net_idx = net_idx
        self.M = num_lines
        if len(vars) != num_lines:
            raise ValueError("Length of consts must be {:d}!".format(num_lines))
        self.vars = vars  # np.ndarray of xi or eta variables
        if len(consts) != 2:
            raise ValueError("Length of consts must be 2!")
        self.consts = consts  # constant lines of xi or eta
        self.var_xe = var_xe  # specifies the changing coord
        # e.g. var_xe == "xi"
        #    xi_0 = vars[0], xi_1 = vars[1], ...
        # ---.--------.-----.------ eta = consts[1]
        #    |        |     |
        # ---.--------.-----.------ eta = consts[0]

        # these include both BC and PDE points
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        self.train_next_batch()
        self.test()

    def losses(self, loss_fn, xei, xef, net, e=False):
        error = self.bc.error(xei, xef, net)
        loss = loss_fn(bkd.zeros_like(error), error)
        if e:
            return loss, error
        else:
            return loss

    def losses_train(self, loss_fn, xei, xef, net, e=False):
        """Return a list of losses for training dataset, i.e., constraints."""
        return self.losses(loss_fn, xei, xef, net, e=e)

    def losses_test(self, loss_fn, xei, xef, net, e=False):
        """Return a list of losses for test dataset, i.e., constraints."""
        return self.losses(loss_fn, xei, xef, net, e=e)

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x = self.train_points()
        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        self.test_x = self.test_points()
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self):
        """Resample the training points."""
        self.train_x, self.train_y, self.train_aux_vars = None, None, None
        self.train_next_batch()

    @run_if_all_none("train_x")
    def train_points(self):
        # only support 2-D case
        X = []
        for const in self.consts:
            tmp = const * np.ones((self.M, 2), dtype=config.real(np))
            if self.var_xe == "xi":
                tmp[:, 0] = self.vars
            else:  # self.var_xe == "eta"
                tmp[:, 1] = self.vars
            X.append(tmp)
        return X

    def test_points(self):
        # TODO: Use different BC points from self.train_x_bc
        x = self.train_x
        return x
