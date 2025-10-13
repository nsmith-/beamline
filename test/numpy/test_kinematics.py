from functools import partial

import hepunits as u
import numpy as np
import pytest
import vector

from beamline.numpy.emfield import EMTensorField, FieldStrength
from beamline.numpy.integrate import solve_ivp
from beamline.numpy.kinematics import (
    ParticleState,
    make_muon,
    ode_tangent_dct,
    ode_tangent_dz,
    polar_tangents,
)
from beamline.units import check_dimensionality, ureg


def test_check_units():
    # FieldStrength
    # "Electric field in [MeV/e/mm]"
    # "Magnetic field in [MeV*ns/e/mm^2]"
    assert check_dimensionality(MeV=1, e=-1, mm=-1, expected=ureg.megavolt / ureg.meter)
    assert check_dimensionality(MeV=1, ns=1, e=-1, mm=-2, expected=ureg.tesla)

    # EM contracted with momentum is a 4-vector in units of [MeV^2 * ns^2 / e / mm^3]
    # so charge/mass * F_{\mu\nu} p^{\nu} has units of force
    assert check_dimensionality(
        MeV=2,
        ns=2,
        e=-1,
        mm=-3,
        expected=ureg.gram / ureg.elementary_charge * ureg.newton,
    )


class _SimpleField(EMTensorField):
    def __init__(self, Ez=0.0, Bz=0.0):
        self.Ez = Ez
        self.Bz = Bz

    def field_strength(self, position: vector.VectorObject4D) -> FieldStrength:
        return FieldStrength(
            E=vector.obj(x=0.0, y=0.0, z=self.Ez),
            B=vector.obj(x=0.0, y=0.0, z=self.Bz),
        )

    def analytic_propagate(self, start: ParticleState, dct: float) -> ParticleState:
        rho: float = start.momentum.to_2D().rho
        if self.Ez == 0.0:
            dz = start.momentum.to_beta3().z * dct
            dpz = 0.0
        else:
            # Distance traveled by a relativistic particle in a constant electric field
            qEzc = start.charge * self.Ez / u.c_light
            a = start.momentum.pz / qEzc
            b2 = (rho**2 + (start.mass * u.c_light) ** 2) / qEzc**2
            dz1 = np.sqrt((a + dct) ** 2 + b2)
            dz2 = np.sqrt(a**2 + b2)
            dz = np.sign(qEzc) * (dz1 - dz2)
            if abs(dz / (dz1 + dz2)) < 1e-5 and False:
                # Avoid catastrophic cancellation
                dz = (
                    a * dct / dz2
                    + b2 * dct**2 / (2 * dz2**3)
                    - a * b2 * dct**3 / (2 * dz2**5)
                )
            dpz = qEzc * dct

        if self.Ez != 0.0 and self.Bz != 0.0 and rho > 0.0:
            raise NotImplementedError(
                "Analytic solution in E field with transverse momentum"
            )
        if self.Bz == 0.0 or rho == 0.0:
            return ParticleState(
                position=start.position + vector.obj(t=dct, x=0.0, y=0.0, z=dz),
                momentum=vector.obj(
                    px=start.momentum.px,
                    py=start.momentum.py,
                    pz=start.momentum.pz + dpz,
                    m=start.momentum.m,
                ),
                mass=start.mass,
                charge=start.charge,
            )
        radius: float = rho * start.charge / self.Bz
        circle_tangent, towards_center = polar_tangents(start.momentum)
        center = start.position - (radius * towards_center).to_xyzt(z=0.0, t=0.0)
        # abs(radius) because omega should be signed by charge
        omega: float = start.momentum.to_beta3().to_2D().dot(circle_tangent) / abs(
            radius
        )
        return ParticleState(
            position=center
            + vector.obj(
                rho=radius,
                phi=towards_center.phi + omega * dct,
                z=start.position.z + dz,
                t=start.position.t + dct,
            ),
            momentum=vector.obj(
                rho=rho,
                phi=circle_tangent.phi + omega * dct,
                pz=start.momentum.pz + dpz,
                m=start.momentum.m,
            ),
            mass=start.mass,
            charge=start.charge,
        )


def _assert_particle_state(end: ParticleState, expected_end: ParticleState):
    assert end.position.t == pytest.approx(expected_end.position.t)
    assert end.position.x == pytest.approx(expected_end.position.x)
    assert end.position.y == pytest.approx(expected_end.position.y)
    assert end.position.z == pytest.approx(expected_end.position.z)
    assert end.momentum.px == pytest.approx(expected_end.momentum.px)
    assert end.momentum.py == pytest.approx(expected_end.momentum.py)
    assert end.momentum.pz == pytest.approx(expected_end.momentum.pz)
    assert end.momentum.energy == pytest.approx(expected_end.momentum.energy)
    assert end.momentum.m == pytest.approx(expected_end.momentum.m)


MeVc = u.MeV / u.c_light
STARTS = {
    "slow_helix_px": make_muon(px=10.0 * MeVc, pz=10.0 * MeVc),
    "slow_helix_py": make_muon(py=10.0 * MeVc, pz=10.0 * MeVc),
    "slow_helix_pxy": make_muon(px=10.0 * MeVc, py=3.0 * MeVc, pz=10.0 * MeVc),
    "slow_straight": make_muon(pz=10.0 * MeVc),
    "fast_helix_px": make_muon(px=10.0 * MeVc, pz=1000.0 * MeVc),
    "fast_helix_py": make_muon(py=10.0 * MeVc, pz=1000.0 * MeVc),
    "fast_helix_pxy": make_muon(px=10.0 * MeVc, py=3.0 * MeVc, pz=1000.0 * MeVc),
    "fast_straight": make_muon(pz=1000.0 * MeVc),
}


@pytest.mark.parametrize("dct", [0.1, 1.0, 10.0], ids=lambda dct: f"dct={dct}m")
@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_ode_tangent_dct_Bz(dct: float, start: ParticleState):
    dct = dct * u.m
    field = _SimpleField(Ez=0.0 * u.megavolt / u.m, Bz=1.0 * u.tesla)
    expected_end = field.analytic_propagate(start, dct)

    sol = solve_ivp(
        fun=partial(ode_tangent_dct, field),
        t_span=(0, dct),
        y0=start,
        rtol=5e-8,  # pytest.approx default is 1e-6, this is what we need for all cases to pass
        atol=1e-12,  # pytest.approx default is 1e-12
    )
    assert sol.success
    end = sol.y[-1]
    assert isinstance(end, ParticleState)
    _assert_particle_state(end, expected_end)


@pytest.mark.parametrize("dct", [0.1, 1.0, 10.0], ids=lambda dct: f"dct={dct}m")
@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_ode_tangent_dz_Bz(dct: float, start: ParticleState):
    dct = dct * u.m
    field = _SimpleField(Ez=0.0 * u.megavolt / u.m, Bz=1.0 * u.tesla)
    expected_end = field.analytic_propagate(start, dct)
    dz = expected_end.position.z

    sol = solve_ivp(
        fun=partial(ode_tangent_dz, field),
        t_span=(0, dz),
        y0=start,
        rtol=2e-8,  # pytest.approx default is 1e-6, this is what we need for all cases to pass
        atol=1e-12,  # pytest.approx default is 1e-12
    )
    assert sol.success

    end = sol.y[-1]
    assert isinstance(end, ParticleState)
    _assert_particle_state(end, expected_end)


STARTS = {
    "slow_forward": make_muon(pz=10.0 * MeVc),
    "fast_forward": make_muon(pz=1000.0 * MeVc),
    "slow_backward": make_muon(pz=-10.0 * MeVc),
    "fast_backward": make_muon(pz=-1000.0 * MeVc),
}


@pytest.mark.parametrize("dct", [0.1, 1.0, 10.0], ids=lambda dct: f"dct={dct}m")
@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_ode_tangent_dct_Ez(dct: float, start: ParticleState):
    dct = dct * u.m
    field = _SimpleField(Ez=0.5 * u.megavolt / u.m, Bz=0.0 * u.tesla)
    expected_end = field.analytic_propagate(start, dct)

    sol = solve_ivp(
        fun=partial(ode_tangent_dct, field),
        t_span=(0, dct),
        y0=start,
        rtol=1e-6,  # pytest.approx default is 1e-6, this is what we need for all cases to pass
        atol=1e-12,  # pytest.approx default is 1e-12
    )
    assert sol.success
    end = sol.y[-1]
    assert isinstance(end, ParticleState)
    _assert_particle_state(end, expected_end)


@pytest.mark.parametrize("dct", [0.1, 1.0, 10.0], ids=lambda dct: f"dct={dct}m")
@pytest.mark.parametrize("start", STARTS.values(), ids=STARTS.keys())
def test_ode_tangent_dz_Ez(dct: float, start: ParticleState):
    dct = dct * u.m
    field = _SimpleField(Ez=0.1 * u.megavolt / u.m, Bz=0.0 * u.tesla)
    expected_end = field.analytic_propagate(start, dct)
    dz = expected_end.position.z

    sol = solve_ivp(
        fun=partial(ode_tangent_dz, field),
        t_span=(0, dz),
        y0=start,
        rtol=1e-6,  # pytest.approx default is 1e-6, this is what we need for all cases to pass
        atol=1e-12,  # pytest.approx default is 1e-12
    )
    assert sol.success

    end = sol.y[-1]
    assert isinstance(end, ParticleState)
    _assert_particle_state(end, expected_end)
