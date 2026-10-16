"""Viewing geometry: rotation curve, observer frame, inclination/position-angle frames, and orient_galaxy."""

import itertools

import numpy as np
import pytest

from tests.synthetic_galaxy import circular_velocity, disk_axes
from vof.calc_rotation_curve import calc_rotation_curve
from vof.orient_galaxy import center_on_observer, orient_galaxy, orientation_vectors


def rotating_disk(n=20000, v_radial=0.0, seed=0):
    """Return (pos, vel, mass) of a thin disk in the z=0 plane following circular_velocity."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.5, 20, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    R_hat = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
    phi_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], axis=1)
    pos = R[:, None] * R_hat
    vel = circular_velocity(R)[:, None] * phi_hat + v_radial * R_hat
    return pos, vel, rng.uniform(1, 2, n)


@pytest.mark.parametrize("v_radial", [0.0, 80.0])
def test_rotation_curve_recovers_circular_velocity(v_radial):
    """Binned speeds match the disk's rotation curve, ignoring radial motion (regression test)."""
    pos, vel, mass = rotating_disk(v_radial=v_radial)
    curve = calc_rotation_curve(pos, vel, mass, nr=20, max_r=20.0)

    edges = np.linspace(0, 20, 21)
    R = np.linalg.norm(pos, axis=1)
    expected = [
        np.average(
            circular_velocity(R[(R >= lo) & (R < hi)]),
            weights=mass[(R >= lo) & (R < hi)],
        )
        for lo, hi in itertools.pairwise(edges[1:])
    ]
    # the first bin is skipped: the disk starts at R=0.5
    np.testing.assert_allclose(curve[1:], expected, rtol=1e-10)


def test_observer_velocity_from_rotation_curve():
    """With no observer velocity given, the observer moves azimuthally at the rotation-curve speed."""
    curve = np.array([0.0, 100.0, 200.0])  # at r = 0, 10, 20 kpc
    pos, vel = center_on_observer(
        [8.0, np.pi / 2, 0.1], None, rotationCurve=curve, max_r=20.0
    )
    np.testing.assert_allclose(pos, [0, 8.0, 0.1], atol=1e-12)
    # phi-hat at phi = 90 deg points along -x
    np.testing.assert_allclose(vel, [-80.0, 0, 0], atol=1e-12)


def test_observer_velocity_given():
    """A given (vr, vphi, vz) velocity is converted to Cartesian at the observer's azimuth."""
    _pos, vel = center_on_observer([-10000, 0, 0], [5.0, 10.0, -3.0])
    np.testing.assert_allclose(vel, [5.0, 10.0, -3.0])


ANGLES = [(i, pa) for i in (0, 30, 60, 90) for pa in (0, 45, 90, 200)]


@pytest.mark.parametrize(("inclination", "position_angle"), ANGLES)
def test_orientation_vectors_inclination(inclination, position_angle):
    """The line of sight (r0_rot) makes the requested inclination with the disk's spin axis."""
    Lhat, r0, _y0 = disk_axes(tilt_deg=30.0)
    Lhat_rot, r0_rot = orientation_vectors(Lhat, r0, inclination, position_angle)

    assert np.linalg.norm(Lhat_rot) == pytest.approx(1.0)
    assert np.linalg.norm(r0_rot) == pytest.approx(1.0)
    assert np.dot(Lhat_rot, r0_rot) == pytest.approx(0.0, abs=1e-12)
    angle = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(Lhat, r0_rot)), np.dot(Lhat, r0_rot))
    )
    assert min(angle, 180 - angle) == pytest.approx(inclination, abs=1e-9)


@pytest.mark.parametrize("position_angle", [0, 45, 90, 200])
def test_orientation_vectors_position_angle(position_angle):
    """position_angle sets the azimuth (about the spin axis) that the line of sight comes from."""
    Lhat, r0, y0 = disk_axes(tilt_deg=30.0)
    _Lhat_rot, r0_rot = orientation_vectors(Lhat, r0, 60, position_angle)
    in_plane = r0_rot - np.dot(r0_rot, Lhat) * Lhat
    azimuth = np.degrees(np.arctan2(np.dot(in_plane, y0), np.dot(in_plane, r0)))
    assert (azimuth - position_angle + 180) % 360 - 180 == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize(("inclination", "position_angle"), ANGLES[4:])  # not face-on
def test_major_axis_is_image_x_axis(inclination, position_angle):
    """The disk's projected major axis always lies along the image's x axis.

    So for a smooth disk every position angle gives the same image, like an observed cube
    rotated so its major axis is horizontal.
    """
    Lhat, r0, _y0 = disk_axes(tilt_deg=30.0)
    Lhat_rot, r0_rot = orientation_vectors(Lhat, r0, inclination, position_angle)
    major_axis = np.cross(
        Lhat, r0_rot
    )  # in the disk plane and perpendicular to the line of sight
    major_axis /= np.linalg.norm(major_axis)
    image_x = np.cross(
        Lhat_rot, r0_rot
    )  # orient_galaxy's y, the image's horizontal axis
    assert abs(np.dot(major_axis, image_x)) == pytest.approx(1.0, abs=1e-12)


def test_orientation_vectors_does_not_modify_inputs():
    """Repeated calls give the same frame and leave Lhat and r0 untouched (regression: r0 accumulated)."""
    Lhat, r0, _y0 = disk_axes()
    Lhat_in, r0_in = Lhat.copy(), r0.copy()
    first = orientation_vectors(Lhat, r0, 45, 90)
    orientation_vectors(Lhat, r0, 45, 90)
    second = orientation_vectors(Lhat, r0, 45, 90)
    np.testing.assert_array_equal(Lhat, Lhat_in)
    np.testing.assert_array_equal(r0, r0_in)
    np.testing.assert_array_equal(first, second)


def test_orient_galaxy_projects_onto_frame():
    """Positions/velocities are expressed along (r0, Lhat x r0, Lhat), and inputs are not modified."""
    Lhat, r0, y0 = disk_axes()
    rng = np.random.default_rng(2)
    pos, vel = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
    pos_in = pos.copy()

    new_pos, new_vel, rot_vel = orient_galaxy(
        pos, vel, Lhat, r0, returnRotationalVelocity=True
    )
    np.testing.assert_allclose(new_pos, pos @ np.stack([r0, y0, Lhat]).T, atol=1e-12)
    np.testing.assert_allclose(new_vel, vel @ np.stack([r0, y0, Lhat]).T, atol=1e-12)
    np.testing.assert_allclose(
        rot_vel, np.cross(pos, vel) @ Lhat / np.linalg.norm(pos, axis=1), atol=1e-12
    )
    np.testing.assert_array_equal(pos, pos_in)
    assert orient_galaxy(pos, None, Lhat, r0).shape == pos.shape
