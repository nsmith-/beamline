from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Stencil:
    stencil: jax.Array
    window_strides: tuple | None
    lhs_dilation: tuple | None
    rhs_dilation: tuple | None
    padding: str | tuple
    eyescale: float | None = None


xy_isolaplace_solver = Stencil(
    stencil=jnp.array(
        [
            [1, 4, 1],
            [4, 0, 4],
            [1, 4, 1],
        ]
    )
    / 20.0,
    window_strides=(1, 1),
    lhs_dilation=None,
    rhs_dilation=None,
    padding=((1, 1), (1, 1)),
    eyescale=6.0 / 20.0,
)


# https://mathematicians.korea.ac.kr/cfdkim/wp-content/uploads/sites/15/2023/06/ISOTROPIC_FDM.pdf
# this is for solving the 3d laplace equation
xyz_isolaplace_solver = Stencil(
    jnp.array(
        [
            [
                [1, 6, 1],
                [6, 20, 6],
                [1, 6, 1],
            ],
            [
                [6, 20, 6],
                [20, 0, 20],
                [6, 20, 6],
            ],
            [
                [1, 6, 1],
                [6, 20, 6],
                [1, 6, 1],
            ],
        ]
    )
    / 200.0,
    window_strides=(1, 1, 1),
    lhs_dilation=None,
    rhs_dilation=None,
    padding=((1, 1), (1, 1), (1, 1)),
    eyescale=48.0 / 200.0,
)


def rp_stencil(ndim: int, prolongate: bool = False):
    """Compute the restriction or prolongation stencil for a given dimension.

    The stencil is a tensor product of 1D stencils.
    """
    scale = 2 if prolongate else 4
    s1 = jnp.array([1, 2, 1]) / scale
    output = s1
    for _ in range(1, ndim):
        output = output[..., None] * s1
    return output[None, None]


def restrict1d(input: jax.Array):
    """Compute the restriction of a 1D array by a factor of 2.

    The input array must be odd in size.
    """
    if input.shape[0] % 2 == 0:
        raise ValueError("Input array must be odd in size.")
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(1),
        window_strides=(2,),
        padding=((1, 1),),
        dimension_numbers=("NCX", "OIX", "NCX"),
    )[0, 0]
    return output


def restrict2d(input: jax.Array):
    """Compute the restriction of a 2D array by a factor of 2.

    The input array must be be odd in size on each axis.
    """
    if len(input.shape) != 2:
        raise ValueError("Input array must be 2D.")
    if input.shape[0] % 2 == 0 or input.shape[1] % 2 == 0:
        raise ValueError("Input array must be odd in size on each axis.")
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(2),
        window_strides=(2, 2),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NCXY", "OIXY", "NCXY"),
    )[0, 0]
    return output


def restrict3d(input: jax.Array):
    """Compute the restriction of a 3D array by a factor of 2.

    The input array must be odd in size on each axis.
    """
    if len(input.shape) != 3:
        raise ValueError("Input array must be 3D.")
    if any(s % 2 == 0 for s in input.shape):
        raise ValueError("Input array must be odd in size on each axis.")
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(3),
        window_strides=(2, 2, 2),
        padding=((1, 1), (1, 1), (1, 1)),
        dimension_numbers=("NCXYZ", "OIXYZ", "NCXYZ"),
    )[0, 0]
    return output


def prolongate1d(input: jax.Array, dim: int = 1):
    """Compute the prolongation of a 1D array by a factor of 2.

    aka linear interpolation

    dim: dimension of parent grid (if we are handling the boundary of a higher-dim grid)
    """
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(1, True),
        window_strides=(1,),
        lhs_dilation=(2,),
        padding=((1, 1),),
        dimension_numbers=("NCX", "OIX", "NCX"),
    )[0, 0]
    boundary_scale = 2 * dim
    output = (
        output.at[1]
        .add(input[0] / boundary_scale)
        .at[-2]
        .add(input[-1] / boundary_scale)
    )
    return output


def prolongate2d(input: jax.Array, dim: int = 2):
    """Compute the prolongation of a 2D array by a factor of 2.

    aka bilinear interpolation

    dim: dimension of parent grid (if we are handling the boundary of a higher-dim grid)
    """
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(2, True),
        window_strides=(1, 1),
        lhs_dilation=(2, 2),
        padding=((1, 1), (1, 1)),
        dimension_numbers=("NCXY", "OIXY", "NCXY"),
    )[0, 0]
    boundary_scale = 2 * (dim - 1)
    output = (
        output.at[1, :]
        .add(prolongate1d(input[0, :], dim) / boundary_scale)
        .at[-2, :]
        .add(prolongate1d(input[-1, :], dim) / boundary_scale)
        .at[:, 1]
        .add(prolongate1d(input[:, 0], dim) / boundary_scale)
        .at[:, -2]
        .add(prolongate1d(input[:, -1], dim) / boundary_scale)
    )
    return output


def prolongate3d(input: jax.Array, dim: int = 3):
    """Compute the prolongation of a 3D array by a factor of 2.

    aka trilinear interpolation

    dim: dimension of parent grid (if we are handling the boundary of a higher-dim grid)
    """
    output = jax.lax.conv_general_dilated(
        input[None, None],
        rp_stencil(3, True),
        window_strides=(1, 1, 1),
        lhs_dilation=(2, 2, 2),
        padding=((1, 1), (1, 1), (1, 1)),
        dimension_numbers=("NCXYZ", "OIXYZ", "NCXYZ"),
    )[0, 0]
    boundary_scale = 2 * (dim - 2)
    output = (
        output.at[1, :, :]
        .add(prolongate2d(input[0, :, :], dim) / boundary_scale)
        .at[-2, :, :]
        .add(prolongate2d(input[-1, :, :], dim) / boundary_scale)
        .at[:, 1, :]
        .add(prolongate2d(input[:, 0, :], dim) / boundary_scale)
        .at[:, -2, :]
        .add(prolongate2d(input[:, -1, :], dim) / boundary_scale)
        .at[:, :, 1]
        .add(prolongate2d(input[:, :, 0], dim) / boundary_scale)
        .at[:, :, -2]
        .add(prolongate2d(input[:, :, -1], dim) / boundary_scale)
    )
    return output
