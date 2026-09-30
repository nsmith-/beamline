"""Shared definitions for the cooling code benchmarks.

Reference:
https://indico.cern.ch/event/1446644/attachments/2918391/5121897/Cooling_Code_Benchmarking-1.pdf
"""

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from matplotlib import pyplot as plt

from beamline.jax.absorber.scattering import highland_scattering_sampler
from beamline.jax.absorber.straggling import landau_energy_loss_sampler
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent, Transform
from beamline.jax.emfield import EMTensorField, SumField, TransformEMField
from beamline.jax.integrate.stochastic import (
    StochasticKick,
    energy_loss_kick,
    scattering_kick,
)
from beamline.jax.kinematics import MuonStateDz
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.rfcavity.pillbox import PillboxCavity
from beamline.jax.types import SBool, SFloat

SOLENOID = ThickSolenoid(
    Rin=250.0 * u.mm,
    Rout=419.3 * u.mm,
    jphi=500.0 * u.A / u.mm**2,
    L=140.0 * u.mm,
)

# Swap these two lines to change straggling/scattering models
STRAGGLING_SAMPLER = landau_energy_loss_sampler
SCATTERING_SAMPLER = highland_scattering_sampler

KICK = StochasticKick(
    straggling=energy_loss_kick(STRAGGLING_SAMPLER),
    scattering=scattering_kick(SCATTERING_SAMPLER),
)

CELL_LENGTH = 800.0 * u.mm
COIL_Z = (100.7 * u.mm, 699.3 * u.mm)  # Table 4 gives the first; second by symmetry
RF_CENTRES = (211.4 * u.mm, 400.0 * u.mm, 588.6 * u.mm)  # 188.6 mm pitch about 400
APERTURE = 81.6 * u.mm  # beam pipe and iris radius are both this
N_FRINGE = 3  # cells added up- and downstream for correct fringe overlap
ABSORBER_HALF = 5.0 * u.mm  # half the Table 5 thickness, one at each end
RF_PHASE_DEG = 20.0  # Table 4, relative to bunching mode
REF_MOMENTUM = 200.0 * u.MeV

def reference_muon() -> MuonStateDz:
    """An on-axis 200 MeV/c muon at the origin; used for mass, energy, beta.
    """
    return MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=REF_MOMENTUM),
        q=1,
    )

class StackedField(EMTensorField):
    """Structurally identical field components, summed under vmap.

    SumField loops in Python, so XLA inlines one copy of each component's code.
    With 14 coils that means 14 copies of ThickSolenoid's Carlson elliptic
    integrals inside the ODE right-hand side, which makes compilation
    intractable. Stacking the components into one pytree and vmapping traces
    the solenoid once and makes the coil count a leading array axis instead.

    Overrides __call__ rather than field_strength because the contraction in
    EMTensorField.__call__ is linear in E and B (so summing outputs equals
    summing fields), and because TransformEMField.field_strength deliberately
    raises.
    """
    stacked: EMTensorField

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        ds = jax.vmap(lambda f: f.signed_time_to_boundary(ray))(self.stacked)
        return ds[jnp.argmin(jnp.abs(ds))]

    def field_strength(self, point: Cartesian4):
        raise RuntimeError("This method should not be used, use __call__ instead")

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        outs = jax.vmap(lambda f: f(vec))(self.stacked)
        return Tangent(p=vec.p, t=Cartesian4(outs.t.coords.sum(axis=0)))


def stack_fields(fields: list[EMTensorField]) -> StackedField:
    """Combine same-shaped fields into one pytree with a leading component axis.
    """
    return StackedField(
        stacked=jax.tree.map(lambda *xs: jnp.stack(jnp.asarray(xs)), *fields)
    )

def cooling_cell_field(phase_deg: float = RF_PHASE_DEG) -> EMTensorField:
    """Coils plus the three-cell RF cavity for the demonstrator cell.

    Coils alternate polarity with an 800 mm period. N_FRINGE cells are added
    up- and downstream so the overlapping fringe fields at the tracking
    boundaries are right (section 4).

    Each cavity's phase is derived from the reference particle's arrival time
    at that cavity's own centre. This reproduces the required 180 deg adjacent
    phasing automatically: the 188.6 mm pitch is beta*lambda/2 = 188.26 mm at
    beta = 0.8842, so consecutive phases differ by very nearly pi. Deriving it
    also absorbs that 0.2% mismatch, which hard-coding 0, pi, 0 would not.

    Sign convention: the reference particle reaches a cavity centre at phase
    argument pi/2 (bunching, no net energy change); subtracting phase_deg moves
    it into the accelerating half.
    """
    frequency = 704.0 * u.MHz
    beta = reference_muon().beta()
    phase_turns = phase_deg / 360.0

    # Coils: one list, stacked, because they are the expensive part to compile.
    coil_pos = SOLENOID
    coil_neg = eqx.tree_at(lambda s: s.jphi, SOLENOID, -SOLENOID.jphi)
    coils = [
        TransformEMField(
            transform=Transform.make_translation(z=zc + n * CELL_LENGTH),
            field=coil,
        )
        for n in range(-N_FRINGE, N_FRINGE + 1)
        for zc, coil in zip(COIL_Z, (coil_pos, coil_neg), strict=True)
    ]

    # Cavities: left unstacked. PillboxCavity has a string field (mode), which
    # jnp.stack cannot handle, and three inlined copies are cheap anyway.
    cavities = []
    for zc in RF_CENTRES:
        tc = zc / (beta * u.c_light)
        phase = -2 * u.pi * ((tc * frequency - 0.25 + phase_turns) % 1.0)
        cavities.append(
            TransformEMField(
                transform=Transform.make_translation(z=zc),
                field=PillboxCavity(
                    length=183.6 * u.mm,
                    frequency=frequency,
                    E0=30.0 * u.MV / u.m,
                    mode="TM",
                    m=0,
                    n=1,
                    p=0,
                    phase=phase,
                ),
            )
        )

    return SumField([stack_fields(coils), *cavities])


def make_beam(
    key, n: int, *, momentum, beta_perp, eps_perp, sigma_t, sigma_E
) -> MuonStateDz:
    """Cylindrically symmetric Gaussian beam with alpha = 0 and L_kin = 0.

        sigma_x  = sqrt(beta_perp * eps_perp * m / p)
        sigma_px = sqrt(eps_perp * m * p / beta_perp)

    With alpha = 0 and L_kin = 0 there are no correlations, so x, px, y, py are
    four independent Gaussians. L_kin is the kinetic angular momentum, fixed
    directly by the x-py / y-px correlation; no vector potential enters.
    pz then follows on-shell from the sampled total energy.

    Targets: Table 4 gives sigma_x = 0.375923 mm, sigma_px = 0.702660 MeV/c;
    Table 5 gives sigma_x = 11.887730 mm, sigma_px = 22.220050 MeV/c.
    """
    ref = MuonStateDz.make(
        position=Cartesian4.make(),
        momentum=Cartesian3.make(z=momentum),
        q=1,
    )
    m, E0 = ref.mass, ref.kin.t.ct

    kx, kpx, ky, kpy, kt, kE = jr.split(key, 6)
    sig_x = jnp.sqrt(beta_perp * eps_perp * m / momentum)
    sig_p = jnp.sqrt(eps_perp * m * momentum / beta_perp)

    x = sig_x * jr.normal(kx, (n,))
    y = sig_x * jr.normal(ky, (n,))
    px = sig_p * jr.normal(kpx, (n,))
    py = sig_p * jr.normal(kpy, (n,))
    E = E0 + sigma_E * jr.normal(kE, (n,))
    ct = u.c_light * sigma_t * jr.normal(kt, (n,))
    pz = jnp.sqrt(E**2 - m**2 - px**2 - py**2)

    # MuonState.make computes the energy via coords.dot(coords), which is only
    # valid for a single 3-vector, so build the ensemble one particle at a time.
    @jax.vmap
    def one(x, y, ct, px, py, pz) -> MuonStateDz:
        return MuonStateDz.make(
            position=Cartesian4.make(x=x, y=y, z=0.0 * u.mm, ct=ct),
            momentum=Cartesian3.make(x=px, y=py, z=pz),
            q=1,
        )

    return one(x, y, ct, px, py, pz)


def optics(track: MuonStateDz, mask, index: int, mass: float) -> dict[str, float]:
    """Optical quantities at save point `index`, over the masked particles.

        eps_perp   = det(Sigma)^(1/4) / m over (x, px, y, py)   [mm]
        beta_perp  = p <(x^2 + y^2)/2> / (m eps_perp)           [mm]
        alpha_perp = -(<x px> + <y py>) / (2 m eps_perp)
        L_kin      = <x py - y px>                              [mm MeV/c]
        eps_long   = sqrt(det Sigma) over (t, E)                [eV ms]

    eps_long is in eV*ms because that is the unit Tables 4 and 5 quote: their
    sigma_t * sigma_E products are 1.304e-3 and 1.3047 in eV*ms.

    The det^(1/4) estimator is biased low for small ensembles so use a
    few thousand particles before reading these as physics.
    """
    sel = np.asarray(mask)
    x = np.asarray(track.kin.p.x[sel, index]) / u.mm
    y = np.asarray(track.kin.p.y[sel, index]) / u.mm
    ct = np.asarray(track.kin.p.ct[sel, index])
    px = np.asarray(track.kin.t.x[sel, index]) / u.MeV
    py = np.asarray(track.kin.t.y[sel, index]) / u.MeV
    pz = np.asarray(track.kin.t.z[sel, index]) / u.MeV
    E = np.asarray(track.kin.t.ct[sel, index]) / u.MeV

    eps_perp = float(np.linalg.det(np.cov(np.stack([x, px, y, py]))) ** 0.25) / mass
    p_mean = float(np.mean(np.sqrt(px**2 + py**2 + pz**2)))
    t_ms = ct / u.c_light / u.ns * 1e-6  # CLHEP length -> ns -> ms
    eps_long = float(
        np.sqrt(np.linalg.det(np.cov(np.stack([t_ms, E * 1e6]))))
    )

    return {
        "eps_perp_mm": eps_perp,
        "beta_perp_mm": p_mean * 0.5 * (x.var() + y.var()) / (mass * eps_perp),
        "alpha_perp": float(
            -0.5 * (np.cov(x, px)[0, 1] + np.cov(y, py)[0, 1]) / (mass * eps_perp)
        ),
        "L_kin_mm_MeV": float(np.mean(x * py - y * px)),
        "eps_long_eV_ms": eps_long,
        "mean_E_MeV": float(E.mean()),
        "sigma_E_MeV": float(E.std()),
        "p_mean_MeV": p_mean,
    }


def write_cell_artifacts(artifacts_dir, stem, zs, track, survived, mass, header):
    """Write optics-vs-z, end-plane profiles, and a four-panel optics figure.

    Returns the per-save-point optics dicts so the caller can assert on them.
    """
    rows = [optics(track, survived, i, mass) for i in range(len(zs))]
    keys = list(rows[0])

    # Optics along the cell: one row per save point.
    with open(artifacts_dir / f"{stem}_optics.csv", "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write("z_mm," + ",".join(keys) + "\n")
        for i, r in enumerate(rows):
            f.write(
                f"{zs[i] / u.mm:.4f}," + ",".join(f"{r[k]:.8g}" for k in keys) + "\n"
            )

    # End-plane profiles, one row per surviving particle. The PDF asks for
    # x, px, y, py, t and kinetic energy.
    sel = np.asarray(survived)
    prof = np.column_stack(
        [
            np.asarray(track.kin.p.x[sel, -1]) / u.mm,
            np.asarray(track.kin.t.x[sel, -1]) / u.MeV,
            np.asarray(track.kin.p.y[sel, -1]) / u.mm,
            np.asarray(track.kin.t.y[sel, -1]) / u.MeV,
            np.asarray(track.kin.p.ct[sel, -1]) / u.c_light / u.ns,
            np.asarray(track.kin.t.ct[sel, -1]) / u.MeV - mass,
        ]
    )
    np.savetxt(
        artifacts_dir / f"{stem}_profiles.csv",
        prof,
        delimiter=",",
        fmt="%.6f",
        header="\n".join([*header, "x_mm,px_MeV,y_mm,py_MeV,t_ns,KE_MeV"]),
        comments="# ",
    )

    zmm = np.asarray(zs) / u.mm
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, lab in zip(
        axes.flat,
        ["beta_perp_mm", "L_kin_mm_MeV", "eps_perp_mm", "eps_long_eV_ms"],
        [
            r"$\beta_\perp$ [mm]",
            r"$L_{kin}$ [mm MeV/c]",
            r"$\epsilon_\perp$ [mm]",
            r"$\epsilon_L$ [eV ms]",
        ],
        strict=True,
    ):
        ax.plot(zmm, [r[key] for r in rows], lw=1.2)
        ax.set(xlabel="z [mm]", ylabel=lab)
        ax.grid(alpha=0.3)
    fig.suptitle(stem.replace("_", " "))
    fig.tight_layout()
    fig.savefig(artifacts_dir / f"{stem}_optics.png", dpi=140)
    plt.close(fig)

    return rows