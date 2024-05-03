import jax
import optax

from coeffset import CoeffSet
from attention_coeffnet import CoeffNet
from torch.utils.data import DataLoader

from jax import numpy as jnp



def mean_squared_error(y_pred, y, mask=1):
    return jnp.mean(optax.l2_loss(y_pred, y))



def train_step(model_apply, params, optimizer_update, opt_state, batch):
    #C_dftb, _, C_delta, pad_mask = batch
    C_dftb, C_delta = batch
    def loss_fn(params):
        pred_C_delta = model_apply(params, C_dftb)#, pad_mask)
        loss = mean_squared_error(pred_C_delta, C_delta)
        return loss, pred_C_delta
    (loss, pred_C_delta), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state



def valid_step(model_apply, params, batch):
    #C_dftb, _, C_delta, pad_mask = batch
    C_dftb, C_delta = batch
    pred_C_delta = model_apply(params, C_dftb)#, pad_mask)
    loss = mean_squared_error(pred_C_delta, C_delta)
    return loss



def train_model(model, train_dataset, valid_dataset, learning_rate, num_epochs, batch_size):
    init_C_dftb, _ = train_dataset.__getitem__(0)
    init_key = jax.random.PRNGKey(0)
    params = model.init(init_key, init_C_dftb)

    '''
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        #collate_fn=lambda batch: train_dataset.pad_collate_fn(batch)
    )
    '''

    '''
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        #collate_fn=lambda batch: valid_dataset.pad_collate_fn(batch)
    )
    '''

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    train_losses = []
    valid_losses = []

    num_train = train_dataset.__len__()
    num_valid = valid_dataset.__len__()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i in range(num_train):
            print(f"Epoch {epoch} Train Sample Nr. {i}")
            batch = train_dataset.__getitem__(i)
            loss, params, opt_state = train_step(model.apply, params, optimizer.update, opt_state, batch)
            train_loss += (loss - train_loss) / (i + 1)
        train_losses.append(train_loss)
        valid_loss = 0.0
        for i in range(num_valid):
            print(f"Epoch {epoch} Valid Sample Nr. {i}")
            batch = valid_dataset.__getitem__(i)
            loss = valid_step(model.apply, params, batch)
            valid_loss += (loss - valid_loss) / (i + 1)
        valid_losses.append(valid_loss)
    
    for i, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
        print(f"Epoch {i}: Train Loss = {train_loss:8.3f}, Valid Loss = {valid_loss:8.3f}")
            

    '''
    for epoch in range(num_epochs):
        print(f"*** EPOCH {epoch+1} ***")
        train_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            print(f"Train Batch Nr. {i}")
            loss, params, opt_state = train_step(model.apply, params, optimizer.update, opt_state, batch)
            train_loss += (loss - train_loss) / (i + 1)
        valid_loss = 0.0
        for i, batch in enumerate(valid_dataloader):
            print(f"Valid Batch Nr. {i}")
            loss = valid_step(model.apply, params, batch)
            valid_loss += (loss - valid_loss) / (i + 1)
        print()
        print(f"epoch: {epoch+1: 3d} train: valid:")
        print(f"loss {train_loss:8.3f} {valid_loss:8.3f}")
        print()
    '''
    
    return params



train_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/annotation.dat"
train_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/DFTB"
train_rose_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/ROSE"
train_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/DELTA"
train_xyz_dir = "/Users/dario/datasets/C_sets/full_molecule/small_train_set/XYZ"

valid_annotation_file = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/annotation.dat"
valid_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/DFTB"
rose_dftb_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/ROSE"
valid_delta_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/datasets/C_sets/full_molecule/small_valid_set/XYZ"

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
    rose_dir=train_rose_dir,
    delta_dir=valid_delta_dir,
    xyz_dir=valid_xyz_dir
)

# hyperparameters
num_features = 4
num_heads = 1
num_attention_blocks = 2

# training parameters
learning_rate = 1e-3
num_epochs = 5
batch_size = 1

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
