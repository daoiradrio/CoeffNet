import jax
import optax
import torch

import numpy as np

from coeffset import CoeffSet
from attention_coeffnet import CoeffNet
from torch.utils.data import DataLoader

from jax import numpy as jnp



def attention_collate_fn(batch):
    batch_C_dftb, _, batch_C_delta, _, _ = zip(*batch)

    batch_size = len(batch_C_dftb)
    max_num_mos = max([C_dftb.shape[0] for C_dftb in batch_C_dftb])
    max_num_atoms = max([C_dftb.shape[1] for C_dftb in batch_C_dftb])

    weight_mask = np.zeros((batch_size, max_num_mos, max_num_mos))
    loss_mask = np.ones((batch_size, max_num_mos, max_num_atoms, 1, 4, 1))

    pad_C_dftb = []
    pad_C_delta = []
    for i, (C_dftb, C_delta) in enumerate(zip(batch_C_dftb, batch_C_delta)):
        num_mos = C_dftb.shape[0]
        num_atoms = C_dftb.shape[1]
        mo_pad_len = max_num_mos - num_mos
        atom_pad_len = max_num_atoms - num_atoms
        weight_mask[i, num_mos:max_num_mos, num_mos:max_num_mos] = -np.inf
        loss_mask[i, num_mos:max_num_mos, num_atoms:max_num_atoms, :, :, :] = 0
        pad_C_dftb.append(
            torch.nn.functional.pad(
                torch.from_numpy(C_dftb),
                (0, 0, 0, 0, 0, 0, 0, atom_pad_len, 0, mo_pad_len),
                value=0
            ).numpy()
        )
        pad_C_delta.append(
            torch.nn.functional.pad(
                torch.from_numpy(C_delta),
                (0, 0, 0, 0, 0, 0, 0, atom_pad_len, 0, mo_pad_len),
                value=0
            ).numpy()
        )
    pad_batch_C_dftb = np.stack(pad_C_dftb)
    pad_batch_C_delta = np.stack(pad_C_delta)

    #return pad_batch_C_dftb, pad_batch_C_delta, weight_mask, loss_mask

    norms = np.array([sum([np.linalg.norm(MO) for MO in C_dftb]) for C_dftb in batch_C_dftb])
    return pad_batch_C_dftb, norms, weight_mask, loss_mask



def mean_squared_error(y_pred, y, loss_mask=1):
    return jnp.sqrt(jnp.mean(optax.l2_loss(y_pred * loss_mask, y)))



#'''
def train_step(model_apply, params, optimizer_update, opt_state, batch):
    #C_dftb, C_delta, weight_mask, loss_mask = batch
    C_dftb, norm, weight_mask, _ = batch
    def loss_fn(params):
        #pred_C_delta = model_apply(params, C_dftb, weight_mask)
        #loss = mean_squared_error(pred_C_delta, C_delta, loss_mask)
        #return loss, pred_C_delta
        pred_norm = model_apply(params, C_dftb, weight_mask)
        loss = mean_squared_error(pred_norm, norm)
        return loss, pred_norm
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state



def valid_step(model_apply, params, batch):
    #C_dftb, C_delta, weight_mask, loss_mask = batch
    #pred_C_delta = model_apply(params, C_dftb, weight_mask)
    #loss = mean_squared_error(pred_C_delta, C_delta, loss_mask)
    C_dftb, norm, weight_mask, _ = batch
    pred_norm = model_apply(params, C_dftb, weight_mask)
    loss = mean_squared_error(pred_norm, norm)
    return loss
#'''



'''
def train_step(model_apply, params, optimizer_update, opt_state, batch):
    C_dftb, _, C_delta, _, _ = batch
    def loss_fn(params):
        pred_C_delta = model_apply(params, C_dftb)
        loss = mean_squared_error(pred_C_delta, C_delta)
        return loss, pred_C_delta
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state



def valid_step(model_apply, params, batch):
    C_dftb, _, C_delta, _, _ = batch
    pred_C_delta = model_apply(params, C_dftb)
    loss = mean_squared_error(pred_C_delta, C_delta)
    return loss
'''



def train_model(model, train_dataset, valid_dataset, learning_rate, num_epochs, batch_size):
    init_C_dftb, _, _, _, _ = train_dataset.__getitem__(0)
    init_key = jax.random.PRNGKey(0)
    params = model.init(init_key, init_C_dftb)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=attention_collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=attention_collate_fn
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    train_losses = []
    valid_losses = []

    '''
    num_train = train_dataset.__len__()
    num_valid = valid_dataset.__len__()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i in range(num_train):
            print(f"Epoch {epoch+1} Train Sample Nr. {i+1}")
            batch = train_dataset.__getitem__(i)
            loss, params, opt_state = train_step(model.apply, params, optimizer.update, opt_state, batch)
            train_loss += (loss - train_loss) / (i + 1)
        train_losses.append(train_loss)
        valid_loss = 0.0
        for i in range(num_valid):
            print(f"Epoch {epoch+1} Valid Sample Nr. {i+1}")
            batch = valid_dataset.__getitem__(i)
            loss = valid_step(model.apply, params, batch)
            valid_loss += (loss - valid_loss) / (i + 1)
        valid_losses.append(valid_loss)
        print()
        print(f"Epoch {epoch+1}: Total Train Loss = {train_loss:8.3f}, TotalValid Loss = {valid_loss:8.3f}")
        print()
    for i, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
        print(f"Epoch {i}: Train Loss = {train_loss:8.3f}, Valid Loss = {valid_loss:8.3f}")
    '''
     
    #'''
    for epoch in range(num_epochs):
        print(f"*** EPOCH {epoch+1} ***")
        train_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            print(f"Train Batch Nr. {i+1}")
            loss, params, opt_state = train_step(model.apply, params, optimizer.update, opt_state, batch)
            train_loss += (loss - train_loss) / (i + 1)
        train_losses.append(train_loss)
        valid_loss = 0.0
        for i, batch in enumerate(valid_dataloader):
            print(f"Valid Batch Nr. {i+1}")
            loss = valid_step(model.apply, params, batch)
            valid_loss += (loss - valid_loss) / (i + 1)
        valid_losses.append(valid_loss)
        print()
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:8.3f}, Valid Loss = {valid_loss:8.3f}")
        print()
    for i, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
        print(f"Epoch {i+1}: Train Loss = {train_loss:8.3f},\tValid Loss = {valid_loss:8.3f}")
    #'''
    
    return params



# small data sets
'''
train_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/molslist.dat"
train_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/DFTB"
train_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/ROSE"
train_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/DELTA"
train_xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"
valid_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/molslist.dat"
valid_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/DFTB"
valid_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/ROSE"
valid_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/preprocessed_QM9/x_small_valid_set"
'''

# medium data sets
#'''
train_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/molslist.dat"
train_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/DFTB"
train_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/ROSE"
train_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_train_set/DELTA"
train_xyz_dir = "/Users/dario/preprocessed_QM9/y_medium_train_set"
valid_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/molslist.dat"
valid_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/DFTB"
valid_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/ROSE"
valid_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/medium_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/preprocessed_QM9/y_medium_valid_set"
#'''

# initialize data loaders
train_dataset = CoeffSet(
    annotation_file=train_annotation_file,
    dftb_dir=train_dftb_dir,
    rose_dir=train_rose_dir,
    delta_dir=train_delta_dir,
    xyz_dir=train_xyz_dir
)

valid_dataset = CoeffSet(
    annotation_file=valid_annotation_file,
    dftb_dir=valid_dftb_dir,
    rose_dir=valid_rose_dir,
    delta_dir=valid_delta_dir,
    xyz_dir=valid_xyz_dir
)

# hyperparameters
num_features = 4
num_heads = 1
num_attention_blocks = 2

# training parameters
learning_rate = 0.001
num_epochs = 5
batch_size = 10

# initialize model
model = CoeffNet(num_features=num_features, num_heads=num_heads, num_blocks=num_attention_blocks)

train_model(model, train_dataset, valid_dataset, learning_rate, num_epochs, batch_size)

'''
Rmat = e3x.so3.random_rotation(jax.random.PRNGKey(0))

temp1 = model.apply(params, init_C_dftb)
res1 = jnp.copy(temp1)
for i_mo in range(num_aos):
    for i_elem, elem in enumerate(elems):
        for i_feature in range(4):
            res1 = res1.at[i_mo, i_elem, 1, 1:, i_feature].set(jnp.dot(Rmat, res1[i_mo, i_elem, 1, 1:, i_feature]))

temp2 = jnp.copy(init_C_dftb)
for i_mo in range(num_aos):
    for i_elem, elem in enumerate(elems):
        for i_feature in range(1):
            temp2 = temp2.at[i_mo, i_elem, 1, 1:, i_feature].set(jnp.dot(Rmat, init_C_dftb[i_mo, i_elem, 1, 1:, i_feature]))
res2 = model.apply(params, temp2)

print(jnp.allclose(res1, res2))
'''

'''
import numpy as np

f = np.random.rand(4)
f = np.expand_dims(f, (-3, -1))
f = jnp.asarray(f)

params = model.init(init_key, f)
Rmat = e3x.so3.random_rotation(jax.random.PRNGKey(0))

res1 = model.apply(params, f)
res1 = res1.at[0, 1:, 0].set(jnp.dot(Rmat, res1[0, 1:, 0]))

f = f.at[0, 1:, 0].set(jnp.dot(Rmat, f[0, 1:, 0]))
res2 = model.apply(params, f)

print(jnp.allclose(res1, res2))
'''
