"""Independent power/phase checks for the absorbing-film backside correction.

Run with the existing optics Python environment; no files are written.
The reference calculation uses characteristic matrices, while the production
implementation uses Airy amplitudes plus an incoherent power series.
"""
import json
import math
import unittest

import numpy as np
import real_data_check as m


METRICS = {}


def admittances(n1, n2, theta, pol):
    u = math.sin(theta) ** 2
    c0, c1, c2 = math.cos(theta), np.sqrt(1-u/n1**2+0j), np.sqrt(1-u/n2**2+0j)
    return (c0, n1*c1, n2*c2) if pol == 0 else (1/c0, n1/c1, n2/c2)


def layer_matrix(y, delta):
    co, si = np.cos(delta), np.sin(delta)
    return np.array([[co, 1j*si/y], [1j*y*si, co]])


def power_from_matrix(matrix, yin, yout):
    b = matrix[0, 0] + matrix[0, 1]*yout
    c = matrix[1, 0] + matrix[1, 1]*yout
    den = yin*b+c
    return abs((yin*b-c)/den)**2, np.real(yout)/np.real(yin)*abs(2*yin/den)**2


def matrix_film(lam, n1, d, n2, theta, pol, reverse=False):
    y0, y1, y2 = admittances(n1, n2, theta, pol)
    delta = 2*np.pi*n1*np.sqrt(1-math.sin(theta)**2/n1**2+0j)*d/lam
    return power_from_matrix(layer_matrix(y1, delta), y2 if reverse else y0, y0 if reverse else y2)


class BacksidePhysicsTests(unittest.TestCase):
    def test_directional_powers_against_independent_matrix(self):
        worst = 0.0
        for k in (0.0, 1e-8, 0.01, 0.2):
            for theta in (0.0, 0.25, 0.6, math.asin(0.9)):
                lam = np.array([440., 695., 950.])
                n1, n2 = 1.9-1j*k, 1.47
                for pol, actual in enumerate(m.film_power_rt(lam, n1, 536.4, n2, theta)):
                    ref = (*matrix_film(lam, n1, 536.4, n2, theta, pol),
                           *matrix_film(lam, n1, 536.4, n2, theta, pol, True))
                    for x, y in zip(actual, ref):
                        worst = max(worst, float(np.max(np.abs(x-y))))
                        np.testing.assert_allclose(x, y, atol=2e-14, rtol=2e-13)
        METRICS['matrix_max_abs_error'] = worst

    def test_conservation_passivity_and_reciprocity(self):
        worst_conservation = 0.0
        for k in (0., 1e-8, 0.01, 0.2):
            for theta in (0., 0.3, math.asin(0.9)):
                for rf, tf, rr, tr in m.film_power_rt(m.LAM_A, 1.9-1j*k, 536.4, m.NS_A, theta):
                    np.testing.assert_allclose(tf, tr, atol=2e-14, rtol=2e-13)
                    for r, t in ((rf, tf), (rr, tr)):
                        self.assertTrue(np.all(r >= 0) and np.all(t >= 0))
                        self.assertLessEqual(float(np.max(r+t)), 1+2e-14)
                        if k == 0:
                            worst_conservation = max(worst_conservation, float(np.max(abs(r+t-1))))
                            np.testing.assert_allclose(r+t, 1, atol=2e-14, rtol=0)
        METRICS['lossless_conservation_max_error'] = worst_conservation

    def test_incoherent_result_matches_coherent_phase_average(self):
        worst = 0.0
        phases = np.arange(4096)*2*np.pi/4096
        for k in (0., 0.01, 0.2):
            for theta in (0., 0.25, math.asin(0.9)):
                lam, n1, n2, d = 695., 1.9-1j*k, 1.47, 536.4
                total = 0.
                for pol, (rf, tf, rr, tr) in enumerate(m.film_power_rt(lam, n1, d, n2, theta)):
                    y0, y1, y2 = admittances(n1, n2, theta, pol)
                    delta = 2*np.pi*n1*np.sqrt(1-math.sin(theta)**2/n1**2+0j)*d/lam
                    mf = layer_matrix(y1, delta)
                    ms = layer_matrix(y2, phases)
                    # Full coherent air / film / finite glass / air stack.
                    full = np.einsum('ij,jkp->ikp', mf, ms)
                    rc, tc = power_from_matrix(full, y0, y0)
                    rb = abs((y2-y0)/(y2+y0))**2
                    expected = rf+tf*rb*tr/(1-rr*rb)
                    worst = max(worst, abs(float(np.mean(rc)-expected)))
                    np.testing.assert_allclose(np.mean(rc), expected, atol=2e-14, rtol=0)
                    self.assertLessEqual(float(np.max(rc+tc)), 1+2e-14)
                    total += .5*expected
                # K=1 makes the production pupil midpoint equal theta.
                if theta <= np.pi/4:
                    got = m.refl(lam, n1, d, n2, math.sin(2*theta), 'theta', True, K=1)
                    np.testing.assert_allclose(got, total, atol=2e-14, rtol=0)
        METRICS['coherent_phase_average_max_error'] = worst

    def test_zero_back_reflection_and_no_backside(self):
        n1 = 1.9-.01j
        lam = m.LAM_A
        # Glass index equal to air makes the remote interface nonreflecting.
        for na in (0., .25, .9):
            front = m.refl(lam, n1, 536.4, 1., na, 'sin2', False)
            back = m.refl(lam, n1, 536.4, 1., na, 'sin2', True)
            np.testing.assert_allclose(front, back, atol=0, rtol=0)

    def test_zero_thickness_reduces_to_bare_incoherent_slab(self):
        for theta in (0., .3):
            total = 0.
            for pol in (0, 1):
                y0, _, y2 = admittances(1.9-.05j, 1.47, theta, pol)
                rb = abs((y0-y2)/(y0+y2))**2
                total += .5*(2*rb/(1+rb))
            got = m.refl(695., 1.9-.05j, 0., 1.47, math.sin(2*theta), 'theta', True, K=1)
            np.testing.assert_allclose(got, total, atol=2e-14, rtol=0)

    def test_lossless_limit_and_continuity(self):
        n1 = 1.9 + 1.2e4*(1/m.LAM_A**2-1/m.LC_A**2)
        for na in (0., .25, .9):
            expected = 0.
            for th, wt in zip(*m.angles(na, 'sin2')):
                for pol, r in enumerate(m.r_film(m.LAM_A, n1, 536.4, m.NS_A, th)):
                    rf = abs(r)**2
                    y0, _, y2 = admittances(n1, m.NS_A, th, pol)
                    rb = abs((y2-y0)/(y2+y0))**2
                    expected += .5*wt*(rf+(1-rf)**2*rb/(1-rf*rb))
            actual = m.refl(m.LAM_A, n1, 536.4, m.NS_A, na, 'sin2', True)
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
            tiny_k = m.refl(m.LAM_A, n1-1e-12j, 536.4, m.NS_A, na, 'sin2', True)
            np.testing.assert_allclose(actual, tiny_k, atol=2e-11, rtol=0)

    def test_units_invariance(self):
        nm = m.refl(m.LAM_A, 1.9-.01j, 536.4, m.NS_A, .25, 'sin2', True)
        um = m.refl(m.LAM_A/1000, 1.9-.01j, .5364, m.NS_A, .25, 'sin2', True)
        np.testing.assert_allclose(nm, um, atol=2e-14, rtol=0)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BacksidePhysicsTests))
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun, 'metrics': METRICS}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
