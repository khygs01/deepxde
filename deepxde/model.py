__all__ = ["LossHistory", "Model", "TrainState"]

import pickle
from collections import OrderedDict

import numpy as np

from . import config
from . import display
from . import gradients as grad
from . import losses as losses_module
from . import metrics as metrics_module
from . import optimizers
from . import utils
from .backend import backend_name, tf, torch, jax, paddle
from .callbacks import CallbackList

import time
import os
import matplotlib.pyplot as plt

if backend_name == "pytorch":
    import torch
    import torch.profiler as profiler


class Model:
    """A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None
        self.metrics = None
        self.external_trainable_variables = []
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        # Backend-dependent attributes
        self.opt = None
        # Tensor or callable
        self.outputs = None
        self.outputs_losses_train = None
        self.outputs_losses_test = None
        self.train_step = None
        if backend_name == "tensorflow.compat.v1":
            self.sess = None
            self.saver = None
        elif backend_name == "pytorch":
            self.lr_scheduler = None
        elif backend_name == "jax":
            self.opt_state = None
            self.params = None

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String name of an optimizer, or a backend optimizer class
                instance.
            lr (float): The learning rate. For L-BFGS, use
                ``dde.optimizers.set_LBFGS_options`` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String name
                of a loss function or a loss function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay (tuple): Name and parameters of decay to the initial learning rate.
                One of the following options:

                - For backend TensorFlow 1.x:

                    - `inverse_time_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                    - `cosine_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

                - For backend TensorFlow 2.x:

                    - `InverseTimeDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: ("inverse time", decay_steps, decay_rate)
                    - `CosineDecay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: ("cosine", decay_steps, alpha)

                - For backend PyTorch:

                    - `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_: ("step", step_size, gamma)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.
            external_trainable_variables: A trainable ``dde.Variable`` object or a list
                of trainable ``dde.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``dde.Variable`` objects are automatically collected.
        """
        print("Compiling model...")
        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        self.losshistory.set_loss_weights(loss_weights)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if backend_name == "tensorflow.compat.v1":
                print(
                    "Warning: For the backend tensorflow.compat.v1, "
                    "`external_trainable_variables` is ignored, and all trainable "
                    "``tf.Variable`` objects are automatically collected."
                )
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        if backend_name == "tensorflow.compat.v1":
            self._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        elif backend_name == "tensorflow":
            self._compile_tensorflow(lr, loss_fn, decay, loss_weights)
        elif backend_name == "pytorch":
            self._compile_pytorch(lr, loss_fn, decay, loss_weights)
        elif backend_name == "jax":
            self._compile_jax(lr, loss_fn, decay, loss_weights)
        elif backend_name == "paddle":
            self._compile_paddle(lr, loss_fn, decay, loss_weights)
        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay, loss_weights):
        """tensorflow.compat.v1"""
        if not self.net.built:
            self.net.build()
        if self.sess is None:
            if config.xla_jit:
                cfg = tf.ConfigProto()
                cfg.graph_options.optimizer_options.global_jit_level = (
                    tf.OptimizerOptions.ON_2
                )
                self.sess = tf.Session(config=cfg)
            else:
                self.sess = tf.Session()
            self.saver = tf.train.Saver(max_to_keep=None)

        def losses(losses_fn):
            # Data losses
            losses = losses_fn(
                self.net.targets, self.net.outputs, loss_fn, self.net.inputs, self
            )
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses.append(tf.losses.get_regularization_loss())
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
            return losses

        losses_train = losses(self.data.losses_train)
        losses_test = losses(self.data.losses_test)
        total_loss = tf.math.reduce_sum(losses_train)

        # Tensors
        self.outputs = self.net.outputs
        self.outputs_losses_train = [self.net.outputs, losses_train]
        self.outputs_losses_test = [self.net.outputs, losses_test]
        self.train_step = optimizers.get(
            total_loss, self.opt_name, learning_rate=lr, decay=decay
        )

    def _compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""

        @tf.function(jit_compile=config.xla_jit)
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
            return outputs_, losses

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

        def train_step_tfp(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """pytorch"""

        def outputs(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
            return self.net(inputs)

        def outputs_losses(training, inputs, targets, losses_fn):
            self.net.train(mode=training)
            if isinstance(inputs, tuple):
                inputs = tuple(
                    map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                )
            else:
                inputs = torch.as_tensor(inputs)
                inputs.requires_grad_()
            outputs_ = self.net(inputs)
            # Data losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= torch.as_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train(inputs, targets):
            return outputs_losses(True, inputs, targets, self.data.losses_train)

        def outputs_losses_test(inputs, targets):
            return outputs_losses(False, inputs, targets, self.data.losses_test)

        # Another way is using per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options,
        # but not all optimizers (such as L-BFGS) support this.
        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        if self.net.regularizer is None:
            self.opt, self.lr_scheduler = optimizers.get(
                trainable_variables, self.opt_name, learning_rate=lr, decay=decay
            )
        else:
            if self.net.regularizer[0] == "l2":
                self.opt, self.lr_scheduler = optimizers.get(
                    trainable_variables,
                    self.opt_name,
                    learning_rate=lr,
                    decay=decay,
                    weight_decay=self.net.regularizer[1],
                )
            else:
                raise NotImplementedError(
                    f"{self.net.regularizer[0]} regularizaiton to be implemented for "
                    "backend pytorch."
                )

        def train_step(inputs, targets):
            def closure():
                losses = outputs_losses_train(inputs, targets)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step

    def _compile_jax(self, lr, loss_fn, decay, loss_weights):
        """jax"""
        # Initialize the network's parameters
        key = jax.random.PRNGKey(config.jax_random_seed)
        self.net.params = self.net.init(key, self.data.test()[0])
        self.params = [self.net.params, self.external_trainable_variables]
        # TODO: learning rate decay
        self.opt = optimizers.get(self.opt_name, learning_rate=lr)
        self.opt_state = self.opt.init(self.params)

        @jax.jit
        def outputs(params, training, inputs):
            return self.net.apply(params, inputs, training=training)

        def outputs_losses(params, training, inputs, targets, losses_fn):
            nn_params, ext_params = params
            # TODO: Add auxiliary vars
            def outputs_fn(inputs):
                return self.net.apply(nn_params, inputs, training=training)

            outputs_ = self.net.apply(nn_params, inputs, training=training)
            # Data losses
            # We use aux so that self.data.losses is a pure function.
            aux = [outputs_fn, ext_params] if ext_params else [outputs_fn]
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self, aux=aux)
            # TODO: Add regularization loss, weighted losses
            if not isinstance(losses, list):
                losses = [losses]
            losses = jax.numpy.asarray(losses)
            return outputs_, losses

        @jax.jit
        def outputs_losses_train(params, inputs, targets):
            return outputs_losses(params, True, inputs, targets, self.data.losses_train)

        @jax.jit
        def outputs_losses_test(params, inputs, targets):
            return outputs_losses(params, False, inputs, targets, self.data.losses_test)

        @jax.jit
        def train_step(params, opt_state, inputs, targets):
            def loss_function(params):
                return jax.numpy.sum(outputs_losses_train(params, inputs, targets)[1])

            grad_fn = jax.grad(loss_function)
            grads = grad_fn(params)
            updates, new_opt_state = self.opt.update(grads, opt_state)
            new_params = optimizers.apply_updates(params, updates)
            return new_params, new_opt_state

        # Pure functions
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step

    def _compile_paddle(self, lr, loss_fn, decay, loss_weights):
        """paddle"""

        def outputs(training, inputs):
            if training:
                self.net.train()
            else:
                self.net.eval()
            with paddle.no_grad():
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        map(lambda x: paddle.to_tensor(x, stop_gradient=False), inputs)
                    )
                else:
                    inputs = paddle.to_tensor(inputs, stop_gradient=False)
                return self.net(inputs)

        def outputs_losses(training, inputs, targets, losses_fn):
            if training:
                self.net.train()
            else:
                self.net.eval()

            if isinstance(inputs, tuple):
                inputs = tuple(
                    map(lambda x: paddle.to_tensor(x, stop_gradient=False), inputs)
                )
            else:
                inputs = paddle.to_tensor(inputs, stop_gradient=False)

            outputs_ = self.net(inputs)
            # Data losses
            if targets is not None:
                targets = paddle.to_tensor(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            # TODO: regularization
            losses = paddle.concat(losses, axis=0)
            # Weighted losses
            if loss_weights is not None:
                losses *= paddle.to_tensor(loss_weights)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return outputs_, losses

        def outputs_losses_train(inputs, targets):
            return outputs_losses(True, inputs, targets, self.data.losses_train)

        def outputs_losses_test(inputs, targets):
            return outputs_losses(False, inputs, targets, self.data.losses_test)

        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        self.opt = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )

        def train_step(inputs, targets):
            losses = outputs_losses_train(inputs, targets)[1]
            total_loss = paddle.sum(losses)
            total_loss.backward()
            self.opt.step()
            self.opt.clear_grad()

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step

    def _outputs(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs)
            return self.sess.run(self.outputs, feed_dict=feed_dict)
        if backend_name in ["tensorflow", "pytorch", "paddle"]:
            outs = self.outputs(training, inputs)
        elif backend_name == "jax":
            outs = self.outputs(self.net.params, training, inputs)
        return utils.to_numpy(outs)

    def _outputs_losses(self, training, inputs, targets, auxiliary_vars):
        if training:
            outputs_losses = self.outputs_losses_train
        else:
            outputs_losses = self.outputs_losses_test
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs, targets, auxiliary_vars)
            return self.sess.run(outputs_losses, feed_dict=feed_dict)
        if backend_name == "tensorflow":
            outs = outputs_losses(inputs, targets, auxiliary_vars)
        elif backend_name == "pytorch":
            # TODO: auxiliary_vars
            self.net.requires_grad_(requires_grad=False)
            outs = outputs_losses(inputs, targets)
            self.net.requires_grad_()
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            outs = outputs_losses(self.params, inputs, targets)
        elif backend_name == "paddle":
            outs = outputs_losses(inputs, targets)
        return utils.to_numpy(outs[0]), utils.to_numpy(outs[1])

    def _train_step(self, inputs, targets, auxiliary_vars):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(True, inputs, targets, auxiliary_vars)
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name in ["pytorch", "paddle"]:
            # TODO: auxiliary_vars
            self.train_step(inputs, targets)
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            self.params, self.opt_state = self.train_step(
                self.params, self.opt_state, inputs, targets
            )
            self.net.params, self.external_trainable_variables = self.params

    @utils.timing
    def train(
        self,
        iterations=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        epochs=None,
    ):
        """Trains the model.

        Args:
            iterations (Integer): Number of iterations to train the model, i.e., number
                of times the network weights are updated.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every (Integer): Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path (String): Path where parameters were previously saved.
            model_save_path (String): Prefix of filenames created for the checkpoint.
            epochs (Integer): Deprecated alias to `iterations`. This will be removed in
                a future version.
        """
        if iterations is None and epochs is not None:
            print(
                "Warning: epochs is deprecated and will be removed in a future version."
                " Use iterations instead."
            )
            iterations = epochs

        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if backend_name == "tensorflow.compat.v1":
            if self.train_state.step == 0:
                print("Initializing variables...")
                self.sess.run(tf.global_variables_initializer())
            else:
                utils.guarantee_initialized_variables(self.sess)

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            if backend_name == "tensorflow.compat.v1":
                self._train_tensorflow_compat_v1_scipy(display_every)
            elif backend_name == "tensorflow":
                self._train_tensorflow_tfp()
            elif backend_name == "pytorch":
                self._train_pytorch_lbfgs()
        else:
            if iterations is None:
                raise ValueError("No iterations for {}.".format(self.opt_name))
            self._train_sgd(iterations, display_every)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(self, iterations, display_every):
        for i in range(iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_tensorflow_compat_v1_scipy(self, display_every):
        def loss_callback(loss_train):
            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0:
                self.train_state.loss_train = loss_train
                self.train_state.loss_test = None
                self.train_state.metrics_test = None
                self.losshistory.append(
                    self.train_state.step, self.train_state.loss_train, None, None
                )
                display.training_display(self.train_state)

        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        feed_dict = self.net.feed_dict(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_step.minimize(
            self.sess,
            feed_dict=feed_dict,
            fetches=[self.outputs_losses_train[1]],
            loss_callback=loss_callback,
        )
        self._test()

    def _train_tensorflow_tfp(self):
        # There is only one optimization step. If using multiple steps with/without
        # previous_optimizer_results, L-BFGS failed to reach a small error. The reason
        # could be that tfp.optimizer.lbfgs_minimize will start from scratch for each
        # call.
        n_iter = 0
        while n_iter < optimizers.LBFGS_options["maxiter"]:
            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            results = self.train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )
            n_iter += results.num_iterations.numpy()
            self.train_state.epoch += results.num_iterations.numpy()
            self.train_state.step += results.num_iterations.numpy()
            self._test()

            if results.converged or results.failed:
                break

    def _train_pytorch_lbfgs(self):
        prev_n_iter = 0
        while prev_n_iter < optimizers.LBFGS_options["maxiter"]:
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            n_iter = self.opt.state_dict()["state"][0]["n_iter"]
            if prev_n_iter == n_iter:
                # Converged
                break

            self.train_state.epoch += n_iter - prev_n_iter
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True
        display.training_display(self.train_state)

    def predict(self, x, operator=None, callbacks=None):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output, otherwise returns the output of the `operator`.

        Args:
            x: The network inputs. A Numpy array or a tuple of Numpy arrays.
            operator: A function takes arguments (`inputs`, `outputs`) or (`inputs`,
                `outputs`, `auxiliary_variables`) and outputs a tensor. `inputs` and
                `outputs` are the network input and output tensors, respectively.
                `auxiliary_variables` is the output of `auxiliary_var_function(x)`
                in `dde.data.PDE`. `operator` is typically chosen as the PDE (used to
                define `dde.data.PDE`) to predict the PDE residual.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during prediction.
        """
        if isinstance(x, tuple):
            x = tuple(np.asarray(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.asarray(x, dtype=config.real(np))
        callbacks = CallbackList(callbacks=callbacks)
        callbacks.set_model(self)
        callbacks.on_predict_begin()

        if operator is None:
            y = self._outputs(False, x)
            callbacks.on_predict_end()
            return y

        # operator is not None
        if utils.get_num_args(operator) == 3:
            aux_vars = self.data.auxiliary_var_fn(x).astype(config.real(np))
        if backend_name == "tensorflow.compat.v1":
            if utils.get_num_args(operator) == 2:
                op = operator(self.net.inputs, self.net.outputs)
                feed_dict = self.net.feed_dict(False, x)
            elif utils.get_num_args(operator) == 3:
                op = operator(
                    self.net.inputs, self.net.outputs, self.net.auxiliary_vars
                )
                feed_dict = self.net.feed_dict(False, x, auxiliary_vars=aux_vars)
            y = self.sess.run(op, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            if utils.get_num_args(operator) == 2:

                @tf.function
                def op(inputs):
                    y = self.net(inputs)
                    return operator(inputs, y)

            elif utils.get_num_args(operator) == 3:

                @tf.function
                def op(inputs):
                    y = self.net(inputs)
                    return operator(inputs, y, aux_vars)

            y = op(x)
            y = utils.to_numpy(y)
        elif backend_name == "pytorch":
            self.net.eval()
            inputs = torch.as_tensor(x)
            inputs.requires_grad_()
            outputs = self.net(inputs)
            if utils.get_num_args(operator) == 2:
                y = operator(inputs, outputs)
            elif utils.get_num_args(operator) == 3:
                # TODO: Pytorch backend Implementation of Auxiliary variables.
                # y = operator(inputs, outputs, torch.as_tensor(aux_vars))
                raise NotImplementedError(
                    "Model.predict() with auxiliary variable hasn't been implemented "
                    "for backend pytorch."
                )
            y = utils.to_numpy(y)
        elif backend_name == "paddle":
            self.net.eval()
            inputs = paddle.to_tensor(x, stop_gradient=False)
            outputs = self.net(inputs)
            if utils.get_num_args(operator) == 2:
                y = operator(inputs, outputs)
            elif utils.get_num_args(operator) == 3:
                # TODO: Paddle backend Implementation of Auxiliary variables.
                # y = operator(inputs, outputs, paddle.to_tensor(aux_vars))
                raise NotImplementedError(
                    "Model.predict() with auxiliary variable hasn't been implemented "
                    "for backend paddle."
                )
            y = utils.to_numpy(y)
        callbacks.on_predict_end()
        return y

    # def evaluate(self, x, y, callbacks=None):
    #     """Returns the loss values & metrics values for the model in test mode."""
    #     raise NotImplementedError(
    #         "Model.evaluate to be implemented. Alternatively, use Model.predict."
    #     )

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        # TODO: backend tensorflow
        if backend_name == "tensorflow.compat.v1":
            destination = OrderedDict()
            variables_names = [v.name for v in tf.global_variables()]
            values = self.sess.run(variables_names)
            for k, v in zip(variables_names, values):
                destination[k] = v
        elif backend_name in ["pytorch", "paddle"]:
            destination = self.net.state_dict()
        else:
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
        return destination

    def save(self, save_path, protocol="backend", verbose=0):
        """Saves all variables to a disk file.

        Args:
            save_path (string): Prefix of filenames to save the model file.
            protocol (string): If `protocol` is "backend", save using the
                backend-specific method.

                - For "tensorflow.compat.v1", use `tf.train.Save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#attributes>`_.
                - For "tensorflow", use `tf.keras.Model.save_weights <https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights>`_.
                - For "pytorch", use `torch.save <https://pytorch.org/docs/stable/generated/torch.save.html>`_.
                - For "paddle", use `paddle.save <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html#cn-api-paddle-framework-io-save>`_.

                If `protocol` is "pickle", save using the Python pickle module. Only the
                protocol "backend" supports ``restore()``.

        Returns:
            string: Path where model is saved.
        """
        # TODO: backend tensorflow
        save_path = f"{save_path}-{self.train_state.epoch}"
        if protocol == "pickle":
            save_path += ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.state_dict(), f)
        elif protocol == "backend":
            if backend_name == "tensorflow.compat.v1":
                save_path += ".ckpt"
                self.saver.save(self.sess, save_path)
            elif backend_name == "tensorflow":
                save_path += ".ckpt"
                self.net.save_weights(save_path)
            elif backend_name == "pytorch":
                save_path += ".pt"
                checkpoint = {
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                }
                torch.save(checkpoint, save_path)
            elif backend_name == "paddle":
                save_path += ".pdparams"
                checkpoint = {
                    "model": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                }
                paddle.save(checkpoint, save_path)
            else:
                raise NotImplementedError(
                    "Model.save() hasn't been implemented for this backend."
                )
        if verbose > 0:
            print(
                "Epoch {}: saving model to {} ...\n".format(
                    self.train_state.epoch, save_path
                )
            )
        return save_path

    def restore(self, save_path, verbose=0):
        """Restore all variables from a disk file.

        Args:
            save_path (string): Path where model was previously saved.
        """
        # TODO: backend tensorflow
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        if backend_name == "tensorflow.compat.v1":
            self.saver.restore(self.sess, save_path)
        elif backend_name == "tensorflow":
            self.net.load_weights(save_path)
        elif backend_name == "pytorch":
            checkpoint = torch.load(save_path)
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        elif backend_name == "paddle":
            checkpoint = paddle.load(save_path)
            self.net.set_state_dict(checkpoint["model"])
            self.opt.set_state_dict(checkpoint["opt"])
        else:
            raise NotImplementedError(
                "Model.restore() hasn't been implemented for this backend."
            )

    def print_model(self):
        """Prints all trainable variables."""
        # TODO: backend tensorflow, pytorch
        if backend_name != "tensorflow.compat.v1":
            raise NotImplementedError(
                "state_dict hasn't been implemented for this backend."
            )
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)


class FSModel(Model):
    """A modified version of ``Model`` for free-surface problems

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        super().__init__(data, net)

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String name of an optimizer, or a backend optimizer class
                instance.
            lr (float): The learning rate. For L-BFGS, use
                ``dde.optimizers.set_LBFGS_options`` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String name
                of a loss function or a loss function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay (tuple): Name and parameters of decay to the initial learning rate.
                One of the following options:

                - For backend TensorFlow 1.x:

                    - `inverse_time_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                    - `cosine_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.
            external_trainable_variables: A trainable ``dde.Variable`` object or a list
                of trainable ``dde.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``dde.Variable`` objects are automatically collected.
        """
        print("Compiling model...")
        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        self.losshistory.set_loss_weights(loss_weights)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if backend_name == "tensorflow.compat.v1":
                print(
                    "Warning: For the backend tensorflow.compat.v1, "
                    "`external_trainable_variables` is ignored, and all trainable "
                    "``tf.Variable`` objects are automatically collected."
                )
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        if backend_name == "tensorflow.compat.v1":
            self._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        else:
            raise NotImplementedError("Currently only tensorflow.compat.v1 supported")
        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay, loss_weights):
        super()._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        self.outputs_xy = self.net.outputs_xy

    def _outputs_xy(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(training, inputs)
            return self.sess.run(self.outputs_xy, feed_dict=feed_dict)
        else:
            raise NotImplementedError("Currently only tensorflow.compat.v1 supported")

    def predict(self, x, operator=None, callbacks=None):
        y0 = super().predict(x, operator, callbacks)
        y1 = self._outputs_xy(False, x)
        return [y0, y1]


class FSModel_with_DD(Model):
    """A modified version of ``Model`` for free-surface problems
       using domain decomposition.

    Args:
        datas: A ``list`` of ``deepxde.data.FSPDE`` instances.
        datas: A ``list`` of ``deepxde.data.SharedBdry`` instances.
        nets: A ``list`` of ``deepxde.nn.NN`` instances.
    """

    def __init__(self, datas, nets, datas_shared=[], datas_qline=[]):
        super().__init__(datas, nets)
        self.nets = (
            self.net if isinstance(nets, (list, tuple)) else [nets]
        )  # order of nets should match that of data
        self.datas = self.data if isinstance(datas, (list, tuple)) else [datas]
        self.datas_shared = (
            datas_shared if isinstance(datas_shared, (list, tuple)) else [datas_shared]
        )
        self.datas_qline = (
            datas_qline if isinstance(datas_qline, (list, tuple)) else [datas_qline]
        )
        self.train_state = TrainState_with_DD()
        self.adaptive_weight = False
        self.initialized = False
        self.scipy_optim_res = None

    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String name of an optimizer, or a backend optimizer class
                instance.
            lr (float): The learning rate. For L-BFGS, use
                ``dde.optimizers.set_LBFGS_options`` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String name
                of a loss function or a loss function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay (tuple): Name and parameters of decay to the initial learning rate.
                One of the following options:

                - For backend TensorFlow 1.x:

                    - `inverse_time_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/inverse_time_decay>`_: ("inverse time", decay_steps, decay_rate)
                    - `cosine_decay <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the `loss_weights` coefficients.
            external_trainable_variables: A trainable ``dde.Variable`` object or a list
                of trainable ``dde.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``dde.Variable`` objects are automatically collected.
        """
        print("Compiling model...")
        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            if backend_name == "tensorflow.compat.v1":
                print(
                    "Warning: For the backend tensorflow.compat.v1, "
                    "`external_trainable_variables` is ignored, and all trainable "
                    "``tf.Variable`` objects are automatically collected."
                )
            if not isinstance(external_trainable_variables, list):
                external_trainable_variables = [external_trainable_variables]
            self.external_trainable_variables = external_trainable_variables

        if backend_name == "tensorflow.compat.v1":
            self._compile_tensorflow_compat_v1(lr, loss_fn, decay, loss_weights)
        elif backend_name == "tensorflow":
            self._compile_tensorflow(lr, loss_fn, decay, loss_weights)
        elif backend_name == "pytorch":
            self._compile_pytorch(lr, loss_fn, decay, loss_weights)
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )
        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay, loss_weights):
        """tensorflow.compat.v1"""
        for net in self.nets:
            if not net.built:
                net.build()
        if loss_weights is not None:
            if loss_weights == "adaptive":
                self.adaptive_weight = True
            else:
                loss_weights = [w for loss_weight in loss_weights for w in loss_weight]
        # make placeholders and tensors for domain boundary BCs
        for i, data_shared in enumerate(self.datas_shared):  # domain boundary loop
            net_idxs = data_shared.net_idxs
            for net in (self.nets[net_idxs[0]], self.nets[net_idxs[1]]):
                x = tf.placeholder(config.real(tf), [None, net.layer_size_xy[0]])
                net.x_db[i] = x
                xy, uvp = net.build_net(x)
                net.xy_db[i] = xy
                net.y_db[i] = uvp
        # make placeholders for qline BCs
        for data_qline in self.datas_qline:
            net = self.nets[data_qline.net_idx]
            xei = tf.placeholder(config.real(tf), [None, net.layer_size_xy[0]])
            xef = tf.placeholder(config.real(tf), [None, net.layer_size_xy[0]])
            net.xei, net.xef = xei, xef
        if self.sess is None:
            if config.xla_jit:
                cfg = tf.ConfigProto()
                cfg.graph_options.optimizer_options.global_jit_level = (
                    tf.OptimizerOptions.ON_2
                )
                self.sess = tf.Session(config=cfg)
            else:
                self.sess = tf.Session()
            self.saver = tf.train.Saver(max_to_keep=None)

        def losses(losses_fns):
            losses_total = []
            errors_total = []
            # compute loss terms for each domain
            for losses_fn, net in zip(losses_fns[0], self.nets):  # domain loop
                # Data losses
                losses, errors = losses_fn(
                    net.targets, net.outputs, loss_fn, net.inputs, self, net=net, e=True
                )
                if not isinstance(losses, list):
                    losses = [losses]
                losses_total += losses
                errors_total += errors
            # compute loss terms for domain boundaries
            for i, (losses_fn, data_shared) in enumerate(
                zip(losses_fns[1], self.datas_shared)
            ):  # domain boundary loop
                net_idxs = data_shared.net_idxs
                net0, net1 = self.nets[net_idxs[0]], self.nets[net_idxs[1]]
                inputs_db, outputs_xy_db, outputs_db = [], [], []
                for net in (net0, net1):
                    inputs_db.append(net.inputs_db[i])
                    outputs_xy_db.append(net.outputs_xy_db[i])
                    outputs_db.append(net.outputs_db[i])
                # Data losses
                losses, errors = losses_fn(
                    loss_fn, inputs_db, outputs_xy_db, outputs_db, e=True
                )
                if not isinstance(losses, list):
                    losses = [losses]
                losses_total += losses
                errors_total += errors
            # compute loss terms for specified flow rates
            for losses_fn, data_qline in zip(losses_fns[2], self.datas_qline):
                net = self.nets[data_qline.net_idx]
                xei, xef = net.xei, net.xef
                loss, error = losses_fn(loss_fn, xei, xef, net, e=True)
                losses_total.append(loss)
                errors_total.append(error)
            # Regularization loss
            for net in self.nets:
                if net.regularizer is not None:
                    losses_total.append(tf.losses.get_regularization_loss())
            losses_total = tf.convert_to_tensor(losses_total)
            return losses_total, errors_total

        losses_fns_train = [
            [data.losses_train for data in self.datas],
            [data.losses_train for data in self.datas_shared],
            [data.losses_train for data in self.datas_qline],
        ]
        losses_fns_test = [
            [data.losses_test for data in self.datas],
            [data.losses_test for data in self.datas_shared],
            [data.losses_test for data in self.datas_qline],
        ]
        losses_train, self.errors = losses(losses_fns_train)
        losses_test, _ = losses(losses_fns_test)
        # Weighted losses
        if loss_weights is not None:
            if self.adaptive_weight:
                loss_weights = [
                    tf.placeholder(config.real(tf), shape=())
                    for _ in range(len(self.errors))
                ]
                self.loss_weights_tensors = loss_weights
                self._compute_ntks_ops()
            losses_train *= loss_weights
            losses_test *= loss_weights
        total_loss = tf.math.reduce_sum(losses_train)

        # Tensors
        self.outputs = [net.outputs for net in self.nets]
        self.outputs_xy = [net.outputs_xy for net in self.nets]
        self.outputs_losses_train = [[net.outputs for net in self.nets], losses_train]
        self.outputs_losses_test = [[net.outputs for net in self.nets], losses_test]
        self.train_step = optimizers.get(
            total_loss, self.opt_name, learning_rate=lr, decay=decay
        )

    def _compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""
        if loss_weights is not None:
            loss_weights = [w for loss_weight in loss_weights for w in loss_weight]

        @tf.function(jit_compile=config.xla_jit)
        def outputs(training, inputs):
            return [
                net(input, training=training)[1]
                for net, input in zip(self.nets, inputs)
            ]

        @tf.function(jit_compile=config.xla_jit)
        def outputs_xy(training, inputs):
            return [
                net(input, training=training)[0]
                for net, input in zip(self.nets, inputs)
            ]

        def outputs_losses(training, inputs, inputs_db, losses_fns):
            losses_total = []
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = [
                net(input, training=training) for net, input in zip(self.nets, inputs)
            ]  # output of net.__call__ is xy, uvp
            outputs_xy_uvp_db_ = [
                (
                    self.nets[data_shared.net_idxs[0]](input_db, training=training),
                    self.nets[data_shared.net_idxs[1]](input_db, training=training),
                )
                for data_shared, input_db in zip(self.datas_shared, inputs_db)
            ]
            # compute loss terms for each domain
            for losses_fn, net, input, (xy, uvp) in zip(
                losses_fns[0], self.nets, inputs, outputs_
            ):  # domain loop
                # Data losses
                losses = losses_fn(None, uvp, loss_fn, input, self, outputs_xy=xy)
                if not isinstance(losses, list):
                    losses = [losses]
                # Regularization loss
                if net.regularizer is not None:
                    losses += [
                        tf.math.reduce_sum(net.losses)
                    ]  # calling keras.layers.Dense adds regularization loss to net.losses
                losses_total += losses
            # compute loss terms for domain boundaries
            for losses_fn, input_db, ((xy0, uvp0), (xy1, uvp1)) in zip(
                losses_fns[1], inputs_db, outputs_xy_uvp_db_
            ):  # domain boundary loop
                # Data losses
                losses = losses_fn(loss_fn, [input_db] * 2, [xy0, xy1], [uvp0, uvp1])
                if not isinstance(losses, list):
                    losses = [losses]
                # Regularization loss
                if net.regularizer is not None:
                    losses.append(tf.losses.get_regularization_loss())
                losses_total += losses
            losses_total = tf.convert_to_tensor(losses_total)
            # Weighted losses
            if loss_weights is not None:
                losses_total *= loss_weights
            return [a[1] for a in outputs_], losses_total

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, inputs_db):
            return outputs_losses(
                True,
                inputs,
                inputs_db,
                [
                    [data.losses_train for data in self.datas],
                    [data.losses_train for data in self.datas_shared],
                ],
            )

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, inputs_db):
            return outputs_losses(
                False,
                inputs,
                inputs_db,
                [
                    [data.losses_test for data in self.datas],
                    [data.losses_test for data in self.datas_shared],
                ],
            )

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, inputs_db):
            # start = time.time()
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, inputs_db)[1]
                total_loss = tf.math.reduce_sum(losses)
            # print("gradient tape", time.time() - start)
            net_trainable_variables = [
                tv for net in self.nets for tv in net.trainable_variables
            ]
            trainable_variables = (
                net_trainable_variables + self.external_trainable_variables
            )
            # print("get variables", time.time() - start)
            grads = tape.gradient(total_loss, trainable_variables)
            # print("tape.gradient", time.time() - start)
            opt.apply_gradients(zip(grads, trainable_variables))
            # print("apply gradients", time.time() - start)

        def train_step_tfp(inputs, inputs_db, previous_optimizer_results=None):
            def build_loss():
                losses = outputs_losses_train(inputs, inputs_db)[1]
                return tf.math.reduce_sum(losses)

            net_trainable_variables = [
                tv for net in self.nets for tv in net.trainable_variables
            ]
            trainable_variables = (
                net_trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_xy = outputs_xy
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )

    def _compile_pytorch(self, lr, loss_fn, decay, loss_weights):
        """pytorch"""

        if loss_weights is not None:
            loss_weights = [w for loss_weight in loss_weights for w in loss_weight]

        def _to_tensor(array):
            if isinstance(array, tuple):
                array = tuple(map(lambda x: torch.as_tensor(x).requires_grad_(), array))
            else:
                array = torch.as_tensor(array)
                array.requires_grad_()
            return array

        def outputs(training, inputs):
            ret = []
            for net, input in zip(self.nets, inputs):
                net.train(mode=training)
                with torch.no_grad():
                    input = _to_tensor(input)
                ret.append(net(input)[1])  # uvp
            return ret

        def outputs_xy(training, inputs):
            ret = []
            for net, input in zip(self.nets, inputs):
                net.train(mode=training)
                with torch.no_grad():
                    input = _to_tensor(input)
                ret.append(net(input)[0])  # xy
            return ret

        def outputs_losses(training, inputs, inputs_db, losses_fns):
            # start = time.time()
            outputs_ = []
            outputs_xy_uvp_db_ = []
            inputs = [_to_tensor(input) for input in inputs]
            inputs_db = [_to_tensor(input_db) for input_db in inputs_db]
            # print("(outputs_losses) prepare inputs", time.time() - start)
            # prepare forward passes
            for net, input in zip(self.nets, inputs):
                net.train(mode=training)
                outputs_.append(net(input))  # xy, uvp
            for data_shared, input_db in zip(self.datas_shared, inputs_db):
                outputs_xy_uvp_db_.append(
                    (
                        self.nets[data_shared.net_idxs[0]](input_db),
                        self.nets[data_shared.net_idxs[1]](input_db),
                    )
                )  # (xy0, uvp0), (xy1, uvp1)
            # print("(outputs_losses) forward passes", time.time() - start)
            losses_total = []
            # compute loss terms for each domain
            for losses_fn, net, input, (xy, uvp) in zip(
                losses_fns[0], self.nets, inputs, outputs_
            ):  # domain loop
                # Data losses
                losses = losses_fn(None, uvp, loss_fn, input, self, outputs_xy=xy)
                if not isinstance(losses, list):
                    losses = [losses]
                losses_total += losses
            # print("(outputs_losses) domain losses", time.time() - start)
            # compute loss terms for domain boundaries
            # i = 0
            for losses_fn, input_db, ((xy0, uvp0), (xy1, uvp1)) in zip(
                losses_fns[1], inputs_db, outputs_xy_uvp_db_
            ):  # domain boundary loop
                # Data losses
                losses = losses_fn(loss_fn, [input_db] * 2, [xy0, xy1], [uvp0, uvp1])
                # print("(outputs_losses) bound{}".format(i), time.time() - start)
                # i += 1
                if not isinstance(losses, list):
                    losses = [losses]
                losses_total += losses
            # print("(outputs_losses) domain boundary losses", time.time() - start)
            losses_total = torch.stack(losses_total)
            # Weighted losses
            if loss_weights is not None:
                losses_total *= torch.as_tensor(loss_weights)
            # print("(outputs_losses) apply weights", time.time() - start)
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return [a[1] for a in outputs_], losses_total

        def outputs_losses_train(inputs, inputs_db):
            return outputs_losses(
                True,
                inputs,
                inputs_db,
                [
                    [data.losses_train for data in self.datas],
                    [data.losses_train for data in self.datas_shared],
                ],
            )

        def outputs_losses_test(inputs, inputs_db):
            return outputs_losses(
                False,
                inputs,
                inputs_db,
                [
                    [data.losses_test for data in self.datas],
                    [data.losses_test for data in self.datas_shared],
                ],
            )

        # Another way is using per-parameter options
        # https://pytorch.org/docs/stable/optim.html#per-parameter-options,
        # but not all optimizers (such as L-BFGS) support this.
        net_parameters = [
            param for net in self.nets for param in list(net.parameters())
        ]
        trainable_variables = net_parameters + self.external_trainable_variables
        if self.net[0].regularizer is None:
            self.opt, self.lr_scheduler = optimizers.get(
                trainable_variables, self.opt_name, learning_rate=lr, decay=decay
            )
        else:
            if self.net[0].regularizer[0] == "l2":
                self.opt, self.lr_scheduler = optimizers.get(
                    trainable_variables,
                    self.opt_name,
                    learning_rate=lr,
                    decay=decay,
                    weight_decay=self.net[0].regularizer[1],
                )
            else:
                raise NotImplementedError(
                    f"{self.net[0].regularizer[0]} regularizaiton to be implemented for "
                    "backend pytorch."
                )

        def train_step(inputs, inputs_db):
            def closure():
                start = time.time()
                losses = outputs_losses_train(inputs, inputs_db)[1]
                # print("(train_step) outputs_losses", time.time() - start)
                total_loss = torch.sum(losses)
                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                # print("(train_step) backward", time.time() - start)
                return total_loss

            start = time.time()
            self.opt.step(closure)
            # print("(train_step) step", time.time() - start)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Callables
        self.outputs = outputs
        self.outputs_xy = outputs_xy
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step

    def _outputs(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.get_feed_dict(training, inputs)
            return self.sess.run(self.outputs, feed_dict=feed_dict)
        elif backend_name in ["tensorflow", "pytorch"]:
            outs = self.outputs(training, inputs)
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )
        return utils.to_numpy(outs)

    def _outputs_xy(self, training, inputs):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.get_feed_dict(training, inputs)
            return self.sess.run(self.outputs_xy, feed_dict=feed_dict)
        elif backend_name in ["tensorflow", "pytorch"]:
            outs = self.outputs_xy(training, inputs)
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )
        return utils.to_numpy(outs)

    def _outputs_losses(self, training, inputs, inputs_db, inputs_qline=None):
        if training:
            outputs_losses = self.outputs_losses_train
        else:
            outputs_losses = self.outputs_losses_test
        if backend_name == "tensorflow.compat.v1":
            adapt_weight = self.adaptive_weight
            feed_dict = self.get_feed_dict(
                training, inputs, inputs_db, inputs_qline, adapt_weight
            )
            return self.sess.run(outputs_losses, feed_dict=feed_dict)
        elif backend_name == "tensorflow":
            outs = outputs_losses(inputs, inputs_db)
        elif backend_name == "pytorch":
            [net.requires_grad_(requires_grad=False) for net in self.nets]
            outs = outputs_losses(inputs, inputs_db)
            [net.requires_grad_() for net in self.nets]
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )
        return utils.to_numpy(outs[0]), utils.to_numpy(outs[1])

    def _train_step(self, inputs, inputs_db, inputs_qline=None):
        if backend_name == "tensorflow.compat.v1":
            adapt_weight = self.adaptive_weight
            feed_dict = self.get_feed_dict(
                True, inputs, inputs_db, inputs_qline, adapt_weight
            )
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name in ["tensorflow", "pytorch"]:
            self.train_step(inputs, inputs_db)
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )

    def _params(self):
        if backend_name == "tensorflow.compat.v1":
            params = []
            for n in self.nets:
                for l_uvp, l_xy in zip(n.layers_uvp, n.layers_xy):
                    try:
                        params.extend(l_uvp[0].weights)
                        params.extend(l_xy[0].weights)
                    except AttributeError:  # no `weights` attribute if tf.identity is used
                        pass
            return params
        else:
            raise NotImplementedError(
                "Currently only tensorflow.compat.v1, tensorflow, and pytorch supported"
            )

    @utils.timing
    def _compute_weights(self, mode=1):
        if backend_name == "tensorflow.compat.v1":
            utils.guarantee_initialized_variables(self.sess)

        # construct feed_dict
        if self.train_state.X_train is None:
            self.train_state.set_data_train(
                self.datas, self.datas_shared, self.datas_qline, self.batch_size
            )
        feed_dict = self.get_feed_dict(
            False,
            self.train_state.X_train,
            self.train_state.X_train_db,
            self.train_state.X_train_qline,
        )

        # evaluate ntks and compute weights from ntks
        np_dtype = config.real(np)
        if mode == 0:
            ntks_ = self.sess.run(self.ntks, feed_dict)
            ntks_tr = [list(map(np.trace, ntk)) for ntk in ntks_]
            eigsums = np.fromiter(map(np.sum, ntks_tr), dtype=np_dtype)
        elif mode == 1:
            ntks_ = self.sess.run(self.ntks, feed_dict)
            eigsums = np.array([np.trace(ntk) for ntk in ntks_], dtype=np_dtype)
        elif mode == 2:
            eigsums = self.sess.run(self.ntks, feed_dict)
            eigsums = np.array(eigsums, dtype=np_dtype)
        eigtotal = np.sum(eigsums)
        small = 1e-10
        print("eigsums =", eigsums)
        weights = eigtotal / (eigsums + small)
        print("weights =", eigsums)
        self.losshistory.set_loss_weights(weights)

    @utils.timing
    def _compute_ntks_ops(self, mode=1):
        from tensorflow.python.ops import array_ops
        from tensorflow.python.ops.parallel_for import control_flow_ops

        def reshape_fn(tensor, num_error_pts):
            num_param = np.prod(tensor.get_shape().as_list()[1:])
            return tf.reshape(tensor, shape=(num_error_pts, num_param))

        def compute_tr_ntk(grads, use_pfor=True, parallel_iterations=None):
            def loop_fn(i):
                y = array_ops.gather(grads, i)
                return tf.reduce_sum(y * y)

            N = array_ops.shape(grads)[0]
            if use_pfor:
                pfor_outputs = control_flow_ops.pfor(
                    loop_fn, N, parallel_iterations=parallel_iterations
                )
            else:
                pfor_outputs = control_flow_ops.for_loop(
                    loop_fn,
                    [grads.dtype],
                    N,
                    parallel_iterations=parallel_iterations,
                )
            return tf.reduce_sum(pfor_outputs)

        # construct feed_dict
        if self.train_state.X_train is None:
            self.train_state.set_data_train(
                self.datas, self.datas_shared, self.datas_qline, self.batch_size
            )
        feed_dict = self.get_feed_dict(
            False,
            self.train_state.X_train,
            self.train_state.X_train_db,
            self.train_state.X_train_qline,
        )

        if backend_name == "tensorflow.compat.v1":
            if self.train_state.step == 0 and not self.initialized:
                print("Initializing variables...")
                self.sess.run(tf.global_variables_initializer())
                self.initialized = True
            else:
                utils.guarantee_initialized_variables(self.sess)

        errors = self.sess.run(self.errors, feed_dict)
        errors_shapes = [e.shape[0] for e in errors]

        dRdps = []
        ntks = []
        # compute dR/dtheta (R: errors, theta: network parameters)
        params = self._params()
        for e, s in zip(self.errors, errors_shapes):
            grads = grad.jacobian_vec(e, params, False)
            reshaped_grads = [reshape_fn(g, s) for g in grads]
            dRdps.append(reshaped_grads)

        for reshaped_grads in dRdps:
            if mode == 0:  # method 0
                ntks.append([tf.matmul(g, tf.transpose(g)) for g in reshaped_grads])

            elif mode == 1:  # method 1
                stacked_grads = tf.concat(reshaped_grads, axis=-1)
                ntks.append(tf.matmul(stacked_grads, tf.transpose(stacked_grads)))

            elif mode == 2:  # method 2
                stacked_grads = tf.concat(reshaped_grads, axis=-1)
                ntks.append(compute_tr_ntk(stacked_grads, False, 10))

        self.ntks = ntks

    @utils.timing
    def train(
        self,
        iterations=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        epochs=None,
        x_tests=None,
        y_tests=None,
        plot_every=None,
        save_plot_every=None,
        plot_save_path=None,
        update_every=None,
    ):
        """Trains the model.

        Args:
            iterations (Integer): Number of iterations to train the model, i.e., number
                of times the network weights are updated.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every (Integer): Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path (String): Path where parameters were previously saved.
            model_save_path (String): Prefix of filenames created for the checkpoint.
            epochs (Integer): Deprecated alias to `iterations`. This will be removed in
                a future version.
        """
        if iterations is None and epochs is not None:
            print(
                "Warning: epochs is deprecated and will be removed in a future version."
                " Use iterations instead."
            )
            iterations = epochs

        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if backend_name == "tensorflow.compat.v1":
            if self.train_state.step == 0 and not self.initialized:
                print("Initializing variables...")
                self.sess.run(tf.global_variables_initializer())
                self.initialized = True
            else:
                utils.guarantee_initialized_variables(self.sess)

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(
            self.datas, self.datas_shared, self.datas_qline, batch_size
        )
        self.train_state.set_data_test(self.datas, self.datas_shared, self.datas_qline)
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            if backend_name == "tensorflow.compat.v1":
                self.scipy_optim_res = self._train_tensorflow_compat_v1_scipy(
                    display_every,
                    x_tests,
                    y_tests,
                    plot_every,
                    save_plot_every,
                    plot_save_path,
                )
            else:
                raise NotImplementedError(
                    "Currently only tensorflow.compat.v1 supported"
                )
        else:
            if iterations is None:
                raise ValueError("No iterations for {}.".format(self.opt_name))
            self._train_sgd(
                iterations,
                display_every,
                x_tests,
                y_tests,
                plot_every,
                save_plot_every,
                plot_save_path,
                update_every,
            )
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(
        self,
        iterations,
        display_every,
        x_tests,
        y_tests,
        plot_every,
        save_plot_every,
        plot_save_path,
        update_every,
    ):
        if plot_save_path and not plot_save_path.endswith("/"):
            plot_save_path += "/"
        if save_plot_every and not plot_save_path:
            plot_save_path = os.getcwd() + "/"
            print(
                "Plot save path not specified. Defaulting to current directory: {}".format(
                    plot_save_path
                )
            )
        # start = time.time()
        for i in range(iterations):
            # if i % 10 == 0:
            # print(i, time.time() - start)
            # start = time.time()
            # print(i)
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                self.datas, self.datas_shared, self.datas_qline, self.batch_size
            )
            if update_every is not None and self.adaptive_weight:
                if self.train_state.step % update_every == 0 or i + 1 == iterations:
                    self._compute_weights()
            self._train_step(
                self.train_state.X_train,
                self.train_state.X_train_db,
                self.train_state.X_train_qline,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()
            if plot_every is not None:
                if self.train_state.step % plot_every == 0 or i + 1 == iterations:
                    utils.plotResult(self, x_tests, y_tests)
            if save_plot_every is not None:
                if self.train_state.step % save_plot_every == 0 or i + 1 == iterations:
                    f, fm = utils.plotResult(self, x_tests, y_tests, show=False)
                    it = str(self.train_state.step).zfill(len(str(int(iterations))))
                    f.savefig(
                        plot_save_path + "{}_uvp.png".format(it),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    fm.savefig(
                        plot_save_path + "{}_xy.png".format(it),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(f)
                    plt.close(fm)

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_tensorflow_compat_v1_scipy(
        self,
        display_every,
        x_tests,
        y_tests,
        plot_every,
        save_plot_every,
        plot_save_path,
    ):
        if plot_save_path and not plot_save_path.endswith("/"):
            plot_save_path += "/"
        if save_plot_every and not plot_save_path:
            plot_save_path = os.getcwd() + "/"
            print(
                "Plot save path not specified. Defaulting to current directory: {}".format(
                    plot_save_path
                )
            )

        def loss_callback(loss_train):
            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0:
                self.train_state.loss_train = loss_train
                self.train_state.loss_test = None
                self.train_state.metrics_test = None
                self.losshistory.append(
                    self.train_state.step, self.train_state.loss_train, None, None
                )
                display.training_display(self.train_state)
            if plot_every is not None:
                if self.train_state.step % plot_every == 0:
                    utils.plotResult(self, x_tests, y_tests)
            if save_plot_every is not None:
                if self.train_state.step % save_plot_every == 0:
                    f, fm = utils.plotResult(self, x_tests, y_tests, show=False)
                    it = str(self.train_state.step)
                    f.savefig(
                        plot_save_path + "{}_uvp.png".format(it),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    fm.savefig(
                        plot_save_path + "{}_xy.png".format(it),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(f)
                    plt.close(fm)

        self.train_state.set_data_train(
            self.datas, self.datas_shared, self.datas_qline, self.batch_size
        )
        adapt_weight = self.adaptive_weight
        feed_dict = self.get_feed_dict(
            True,
            self.train_state.X_train,
            self.train_state.X_train_db,
            self.train_state.X_train_qline,
            adapt_weight,
        )
        res = self.train_step.minimize(
            self.sess,
            feed_dict=feed_dict,
            fetches=[self.outputs_losses_train[1]],
            loss_callback=loss_callback,
        )
        self._test()
        return res

    def _test(self):
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.X_train_db,
            self.train_state.X_train_qline,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.X_test_db,
            self.train_state.X_train_qline,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True
        display.training_display(self.train_state)

    def get_feed_dict(
        self, training, inputs, inputs_db=None, inputs_qline=None, adapt_weight=False
    ):
        feed_dict = dict()
        for net, input in zip(self.nets, inputs):
            feed_dict.update(net.feed_dict(training, input))
        if inputs_db is not None:
            # add domain boundary terms to feed_dict
            for i, (data_shared, input_db) in enumerate(
                zip(self.datas_shared, inputs_db)
            ):  # domain boundary loop
                net_idxs = data_shared.net_idxs
                for net in (self.nets[net_idxs[0]], self.nets[net_idxs[1]]):
                    feed_dict.update(net._feed_dict_inputs_db(input_db, i))
        if inputs_qline is not None:
            # add qline boundary inputs to feed_dict
            for (xei, xef), data_qline in zip(inputs_qline, self.datas_qline):
                net = self.nets[data_qline.net_idx]
                feed_dict.update(net._feed_dict_inputs_qline(xei, xef))
        if adapt_weight:
            if self.losshistory.loss_weights is None:
                self._compute_weights()
            feed_dict.update(
                dict(zip(self.loss_weights_tensors, self.losshistory.loss_weights))
            )
        return feed_dict

    def predict(self, xs, callbacks=None):
        """Generates predictions for the input samples. If `operator` is ``None``,
        returns the network output.
        """
        xs = tuple(np.asarray(xi, dtype=config.real(np)) for xi in xs)
        callbacks = CallbackList(callbacks=callbacks)
        callbacks.set_model(self)
        callbacks.on_predict_begin()
        uvp = self._outputs(False, xs)
        xy = self._outputs_xy(False, xs)
        callbacks.on_predict_end()
        return uvp, xy


class TrainState:
    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.train_aux_vars = None
        self.X_test = None
        self.y_test = None
        self.test_aux_vars = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf


class TrainState_with_DD(TrainState):
    def __init__(self):
        super().__init__()
        self.X_train_db = None
        self.X_test_db = None
        self.X_train_qline = None
        self.X_test_qline = None

    def set_data_train(self, datas, datas_shared, datas_qline, batch_size):
        X_train, y_train, train_aux_vars = [], [], []
        for data in datas:
            train_x, train_y, train_aux_var = data.train_next_batch(batch_size)
            X_train.append(train_x)
            y_train.append(train_y)
            train_aux_vars.append(train_aux_var)

        X_train_db = []
        for data in datas_shared:
            train_x, _, _ = data.train_next_batch(batch_size)
            X_train_db.append(train_x)

        X_train_qline = []
        for data in datas_qline:
            train_x, _, _ = data.train_next_batch(batch_size)
            X_train_qline.append(train_x)

        self.X_train = X_train
        self.X_train_db = X_train_db
        self.X_train_qline = X_train_qline
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, datas, datas_shared, datas_qline):
        X_test, y_test, test_aux_vars = [], [], []
        for data in datas:
            test_x, test_y, test_aux_var = data.test()
            X_test.append(test_x)
            y_test.append(test_y)
            test_aux_vars.append(test_aux_var)

        X_test_db = []
        for data in datas_shared:
            test_x, _, _ = data.test()
            X_test_db.append(test_x)

        X_test_qline = []
        for data in datas_qline:
            test_x, _, _ = data.test()
            X_test_qline.append(test_x)

        self.X_test = X_test
        self.X_test_db = X_test_db
        self.X_test_qline = X_test_qline
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars


class LossHistory:
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights_record = []
        self.loss_weights = None

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
        self.loss_weights_record.append(self.loss_weights)
