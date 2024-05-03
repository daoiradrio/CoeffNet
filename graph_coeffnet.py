import e3x
from flax import linen as nn



class CoeffNet(nn.Module):

    num_features: int
    num_refinements: int
    num_basis_funcs: int
    max_degree: int

    @nn.compact
    def __call__(self, x_dftb, coords, dst_idx, src_idx):
        dist_basis = e3x.nn.basis(
            r=e3x.ops.gather_dst(coords, dst_idx=dst_idx) - e3x.ops.gather_src(coords, src_idx=src_idx),
            num=self.num_basis_funcs,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.basic_gaussian
        )
        x_dftb = e3x.nn.TensorDense(features=self.num_features, max_degree=2)(x_dftb)
        x_dftb = x_dftb.reshape((-1, *x_dftb.shape[-3:]))
        x_dftb = x_dftb[:dist_basis.shape[0], :, :, :]
        print()
        print(x_dftb.shape)
        print(dist_basis.shape)
        print()
        for _ in range(self.num_refinements):
            x_refine = e3x.nn.MessagePass(use_basis_bias=True)(x_dftb, dist_basis, dst_idx=dst_idx, src_idx=src_idx)
            x_dftb = e3x.nn.add(x_dftb, x_refine)
        x_dftb = e3x.nn.TensorDense()(x_dftb)
        x_dftb = e3x.nn.relu(x_dftb)
        x_dftb = e3x.nn.TensorDense(features=1, include_pseudotensors=False)(x_dftb)
        return x_dftb
