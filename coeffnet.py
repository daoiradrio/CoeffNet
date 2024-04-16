import e3x
from flax import linen as nn
from jax import numpy as jnp



class CoeffNet(nn.Module):

    n_features: int
    n_refinements: int
    n_basis_funcs: int
    max_degree: int

    @nn.compact
    def __call__(self, x_dftb, coords, dst_idx, src_idx):
        dist_basis = e3x.nn.basis(
            r=e3x.ops.gather_dst(coords, dst_idx=dst_idx) - e3x.ops.gather_src(coords, src_idx=src_idx),
            num=self.n_basis_funcs,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.basic_gaussian
        )
        x_dftb = e3x.nn.TensorDense(features=self.n_features)(x_dftb)
        for i in range(self.n_refinements):
            x_refine = e3x.nn.MessagePass(use_basis_bias=True)(x_dftb, dist_basis, dst_idx=dst_idx, src_idx=src_idx)
            #x_refine = e3x.nn.add(x_dftb, x_refine)
            #x_refine = e3x.nn.Dense(self.n_features)(x_refine)
            #x_refine = e3x.nn.relu(x_refine)
            #x_refine = e3x.nn.Dense(self.n_features)(x_refine)
            x_dftb = e3x.nn.add(x_dftb, x_refine)
        pred_y_delta = e3x.nn.TensorDense(features=1)(x_dftb)
        return pred_y_delta
