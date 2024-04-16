import jax
import optax
import torch
import e3x
import functools
import orbax.checkpoint

import numpy as np

from jax import numpy as jnp
from flax import serialization
from NeuralNetworks.coeffset import CoeffSet
from NeuralNetworks.coeffnet import CoeffNet
from torch.utils.data import DataLoader



def custom_collate_fn(batch):
    tuple_x_dftb, tuple_y_delta, tuple_coords = zip(*batch)
    prev_n_atoms = tuple_coords[0].shape[0]
    batch_dst_idx, batch_src_idx = e3x.ops.sparse_pairwise_indices(prev_n_atoms)
    for coords in tuple_coords[1:]:
        n_atoms = coords.shape[0]
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n_atoms)
        dst_idx += prev_n_atoms
        src_idx += prev_n_atoms
        batch_dst_idx = jnp.concatenate((batch_dst_idx, dst_idx))
        batch_src_idx = jnp.concatenate((batch_src_idx, src_idx))
        prev_n_atoms = n_atoms
    return jnp.vstack(tuple_x_dftb), jnp.vstack(tuple_y_delta), jnp.vstack(tuple_coords), batch_dst_idx, batch_src_idx


def mean_squared_error(y_pred, y):
    return jnp.mean(optax.l2_loss(y_pred, y))


def mean_absolute_error(pred, target):
    return jnp.mean(jnp.abs(pred - target))


@functools.partial(jax.jit, static_argnames=("model_apply"))
def valid_step(model_apply, params, batch):
    x_dftb, y_delta, coords, dst_idx, src_idx = batch
    pred_y_delta = model_apply(params, x_dftb, coords, dst_idx, src_idx)
    loss = mean_squared_error(pred_y_delta, y_delta)
    mae = mean_absolute_error(pred_y_delta, y_delta)
    return loss, mae


@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update"))
def train_step(model_apply, params, optimizer_update, opt_state, batch):
    x_dftb, y_delta, coords, dst_idx, src_idx = batch
    def loss_fn(params):
        pred_y_delta = model_apply(params, x_dftb, coords, dst_idx, src_idx)
        loss = mean_squared_error(pred_y_delta, y_delta)
        return loss, pred_y_delta
    (loss, pred_y_delta), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    mae = mean_absolute_error(pred_y_delta, y_delta)
    return loss, mae, params, opt_state


def train_model(
    model, train_dataset, valid_dataset, n_epochs, learning_rate, batch_size
):
    init_x_dftb, init_y_delta, init_coords = train_dataset.__getitem__(0)
    init_dst_idx, init_src_idx = e3x.ops.sparse_pairwise_indices(init_coords.shape[0])
    init_key = jax.random.PRNGKey(0)
    params = model.init(init_key, init_x_dftb, init_coords, init_dst_idx, init_src_idx)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    for epoch in range(n_epochs):
        print(f"*** EPOCH {epoch+1} ***")
        train_loss = 0
        train_mae = 0
        for i, batch in enumerate(train_dataloader):
            print(f"Train Batch Nr. {i}")
            loss, mae, params, opt_state = train_step(model.apply, params, optimizer.update, opt_state, batch)
            train_loss += (loss - train_loss) / (i+1)
            train_mae += (mae - train_mae) / (i+1)
        valid_loss = 0
        valid_mae = 0
        for i, batch in enumerate(valid_dataloader):
            print(f"Valid Batch Nr. {i}")
            loss, mae = valid_step(model.apply, params, batch)
            valid_loss += (loss - valid_loss) / (i+1)
            valid_mae += (mae - valid_mae) / (i+1)
        print()
        print(f"epoch: {epoch+1: 3d} train: valid:")
        print(f"loss {train_loss:8.3f} {valid_loss:8.3f}")
        print(f"mae {train_mae:8.3f} {valid_mae:8.3f}")
        print()
    
    return params
            


train_annotation_file = "/Users/dario/datasets/C_sets/tiny_train_set/annotation.dat"
train_dftb_dir = "/Users/dario/datasets/C_sets/tiny_train_set/DFTB"
train_delta_dir = "/Users/dario/datasets/C_sets/tiny_train_set/DELTA"
train_xyz_dir = "/Users/dario/datasets/C_sets/tiny_train_set/XYZ"

valid_annotation_file = "/Users/dario/datasets/C_sets/tiny_valid_set/annotation.dat"
valid_dftb_dir = "/Users/dario/datasets/C_sets/tiny_valid_set/DFTB"
valid_delta_dir = "/Users/dario/datasets/C_sets/tiny_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/datasets/C_sets/tiny_valid_set/XYZ"

train_dataset = CoeffSet(
    annotation_file=train_annotation_file,
    dftb_dir=train_dftb_dir,
    delta_dir=train_delta_dir,
    xyz_dir=train_xyz_dir
)

valid_dataset = CoeffSet(
    annotation_file=valid_annotation_file,
    dftb_dir=valid_dftb_dir,
    delta_dir=valid_delta_dir,
    xyz_dir=valid_xyz_dir
)

learning_rate = 0.01
n_epochs = 4
batch_size = 10
coeffmodel = CoeffNet(n_features=4, n_refinements=3, n_basis_funcs=4, max_degree=1)
trained_params = train_model(coeffmodel, train_dataset, valid_dataset, n_epochs, learning_rate, batch_size)

bytes_trained_params = serialization.to_bytes(trained_params)
with open("trained_coeffnet_params.txt", "wb") as outfile:
    outfile.write(bytes_trained_params)
