import numpy as np

from .data import Data
from .. import backend as bkd
from .. import config
from ..backend import backend_name
from ..utils import get_num_args, run_if_all_none

import time


class SharedBdry(Data):
    """Compatible with ``FSModel_with_DD`` class"""

    def __init__(
        self,
        geometry,
        net_idxs,
        bcs,
        num_boundary=0,
        train_distribution="Hammersley",
        anchors=None,
        exclusions=None,
    ):
        self.geom = geometry
        self.net_idxs = net_idxs
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]

        self.num_boundary = num_boundary
        self.train_distribution = train_distribution
        self.anchors = None if anchors is None else anchors.astype(config.real(np))
        self.exclusions = exclusions

        # TODO: train_x_all is used for PDE losses. It is better to add train_x_pde
        # explicitly.
        self.train_x_all = None
        self.train_x_bc = None
        self.num_bcs = None

        # these include both BC and PDE points
        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        self.train_next_batch()
        self.test()

    def losses(self, loss_fn, inputs_db, outputs_xy_db, outputs_db, e=False):
        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * len(self.bcs)
        elif len(loss_fn) != len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(self.bcs), len(loss_fn)
                )
            )

        bcs_start = np.cumsum([0] + self.num_bcs)
        bcs_start = list(map(int, bcs_start))
        losses = []
        errors = []
        # start = time.time()
        for i, bc in enumerate(self.bcs):
            beg, end = bcs_start[i], bcs_start[i + 1]
            # The same BC points are used for training and testing.
            error = bc.error(
                self.train_x,  # X
                inputs_db[0],  # xieta (net0)
                inputs_db[1],  # net 1
                outputs_xy_db[0],  # xy
                outputs_xy_db[1],
                outputs_db[0],  # uvp
                outputs_db[1],
                beg,
                end,
            )
            # print("(shared_bdry) error{}".format(i), time.time() - start)
            errors.append(error)
            losses.append(loss_fn[i](bkd.zeros_like(error), error))
        if e:
            return losses, errors
        else:
            return losses

    def losses_train(self, loss_fn, inputs_db, outputs_xy_db, outputs_db, e=False):
        """Return a list of losses for training dataset, i.e., constraints."""
        return self.losses(loss_fn, inputs_db, outputs_xy_db, outputs_db, e=e)

    def losses_test(self, loss_fn, inputs_db, outputs_xy_db, outputs_db, e=False):
        """Return a list of losses for test dataset, i.e., constraints."""
        return self.losses(loss_fn, inputs_db, outputs_xy_db, outputs_db, e=e)

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = self.train_points()
        self.train_x = self.bc_points()
        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        self.test_x = self.test_points()
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self, pde_points=True, bc_points=True):
        """Resample the training points for PDE and/or BC."""
        if pde_points:
            self.train_x_all = None
        if bc_points:
            self.train_x_bc = None
        self.train_x, self.train_y, self.train_aux_vars = None, None, None
        self.train_next_batch()

    def add_anchors(self, anchors):
        """Add new points for training PDE losses. The BC points will not be updated."""
        anchors = anchors.astype(config.real(np))
        if self.anchors is None:
            self.anchors = anchors
        else:
            self.anchors = np.vstack((anchors, self.anchors))
        self.train_x_all = np.vstack((anchors, self.train_x_all))
        self.train_x = self.bc_points()

    def replace_with_anchors(self, anchors):
        """Replace the current PDE training points with anchors. The BC points will not be changed."""
        self.anchors = anchors.astype(config.real(np))
        self.train_x_all = self.anchors
        self.train_x = self.bc_points()

    @run_if_all_none("train_x_all")
    def train_points(self):
        X = np.empty((0, self.geom.dim), dtype=config.real(np))
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        if self.exclusions is not None:

            def is_not_excluded(x):
                return not np.any([np.allclose(x, y) for y in self.exclusions])

            X = np.array(list(filter(is_not_excluded, X)))
        self.train_x_all = X
        return X

    @run_if_all_none("train_x_bc")
    def bc_points(self):
        x_bcs = [bc.collocation_points(self.train_x_all) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (
            np.vstack(x_bcs)
            if x_bcs
            else np.empty([0, self.train_x_all.shape[-1]], dtype=config.real(np))
        )
        return self.train_x_bc

    def test_points(self):
        # TODO: Use different BC points from self.train_x_bc
        x = self.train_x_bc
        return x
