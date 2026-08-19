"""
Laplace Transform Numerical Inversion Methods.

Two methods implemented from MATLAB originals:
1. Stehfest (1970): Real-valued, N=14 terms, simple but less accurate for oscillatory functions
2. Abate-Whitt (1995): Complex-valued Euler acceleration, more accurate for general transforms

Both methods invert F_hat(p) to f(t) where F_hat(p) = integral_0^inf e^{-pt} f(t) dt.

Usage:
    g: callable, g(p_array) -> array of shape (len(p_array), Nx)
       where Nx is the number of spatial points evaluated simultaneously.
       For scalar output, g(p_array) -> array of shape (len(p_array), 1) or (len(p_array),)
    t: float, the time point at which to invert

Returns:
    f: array of shape (Nx,) — the inverted values at time t
"""
import numpy as np
from math import comb


def laplace_invert_stehfest(g, t, N=14):
    """
    Stehfest algorithm for numerical Laplace inversion.
    
    Real-valued method using N evaluation points along the real axis.
    Best for smooth, non-oscillatory transforms.
    
    Parameters
    ----------
    g : callable
        Laplace-domain function. g(p_array) returns array of shape (len(p_array), Nx).
    t : float
        Time point for inversion. Must be > 0.
    N : int
        Number of terms. Must be even. Default 14. Options: 10, 14, 20.
    
    Returns
    -------
    f : ndarray of shape (Nx,)
    """
    # Pre-computed Stehfest weights for standard N values
    if N == 10:
        Q = np.array([
            0.083333333333333300,
            -32.083333333333300000,
            1279.000000000000000000,
            -15623.666666666700000000,
            84244.166666666700000000,
            -236957.500000000000000000,
            375911.666666667000000000,
            -340071.666666667000000000,
            164062.500000000000000000,
            -32812.500000000000000000,
        ])
    elif N == 14:
        Q = np.array([
            0.002777777777777780,
            -6.402777777777780000,
            924.050000000000000000,
            -34597.927777777800000000,
            540321.111111111000000000,
            -4398346.366666670000000000,
            21087591.777777800000000000,
            -63944913.044444400000000000,
            127597579.550000000000000000,
            -170137188.083333000000000000,
            150327467.033333000000000000,
            -84592161.499999900000000000,
            27478884.766666600000000000,
            -3925554.966666660000000000,
        ])
    elif N == 20:
        Q = np.array([
            -1 / 181440,
            27649 / 181440,
            -98671 / 840,
            131108917 / 7560,
            -2790568153 / 3024,
            119821402447 / 5040,
            -377374859491 / 1080,
            24504756082873 / 7560,
            -68130546312319 / 3360,
            1.62324987232298e15 / 18144,
            -6.50963449161629e15 / 22680,
            1.72113986590941e15 / 2520,
            -1.84325248304221e16 / 15120,
            3.53715940981875e15 / 2160,
            -830177453365403 / 504,
            1.84754992631967e15 / 1512,
            -3.92398206772844e15 / 6048,
            470366372878831 / 2016,
            -28868125000000 / 567,
            2886812500000 / 567,
        ])
    else:
        raise ValueError(f"N must be 10, 14, or 20, got {N}")

    # Evaluation points: p_k = k * ln(2) / t
    s = np.log(2) / t
    transform_points = s * np.arange(1, N + 1)

    # Evaluate the Laplace-domain function at all points
    gp = np.real(g(transform_points))

    # Handle 1D output
    if gp.ndim == 1:
        gp = gp.reshape(-1, 1)

    Nx = gp.shape[1]
    f = np.zeros(Nx)

    for k in range(N):
        f += s * Q[k] * gp[k, :]

    return f


def laplace_invert_abate_whitt(g, t, N=25, M=12, tol=1e-8):
    """
    Abate-Whitt Euler acceleration method for Laplace inversion.
    
    Complex-valued method using the Bromwich integral with Euler summation
    acceleration. More accurate than Stehfest for general transforms,
    especially oscillatory or discontinuous functions.
    
    Reference: Abate, J. and Whitt, W. (1995). Numerical Inversion of Laplace
    Transforms of Probability Distributions. ORSA Journal on Computing 7(1):36-43.
    
    Parameters
    ----------
    g : callable
        Laplace-domain function. g(p_array) returns array of shape (len(p_array), Nx).
        Must handle complex-valued arguments.
    t : float
        Time point for inversion. Must be > 0.
    N : int
        Number of terms in the main sum. Default 25.
    M : int
        Number of Euler acceleration terms. Default 12.
    tol : float
        Tolerance parameter controlling the Bromwich contour shift. Default 1e-8.
    
    Returns
    -------
    f : ndarray of shape (Nx,)
    """
    A = -np.log(tol)
    exp_At = np.exp(0.5 * A) / t

    # Euler acceleration weights: c_k = C(M,k) * 0.5^M
    p_half = 0.5 ** M
    c = np.array([p_half * comb(M, k) for k in range(1, M + 1)])

    # Imaginary step
    h = 1j * np.pi / t
    # Real shift
    x = 0.5 * A / t

    # Build evaluation points: x, x+h, x+2h, ..., x+(N+M)*h
    n_total = 1 + N + M
    transform_points = np.zeros(n_total, dtype=complex)
    transform_points[0] = x
    for k in range(1, n_total):
        transform_points[k] = x + k * h

    # Evaluate the Laplace-domain function at all points
    gp = g(transform_points)

    # Handle 1D output
    if gp.ndim == 1:
        gp = gp.reshape(-1, 1)

    Nx = gp.shape[1]
    f = np.zeros(Nx)

    for nx in range(Nx):
        # Main alternating sum (first 1+N terms)
        s = 0.5 * np.real(gp[0, nx])
        sign = -1
        for k in range(1, N + 1):
            s += sign * np.real(gp[k, nx])
            sign = -sign

        # Euler acceleration (next M terms)
        est = p_half * s
        for k in range(M):
            s += sign * np.real(gp[N + 1 + k, nx])
            est += c[k] * s
            sign = -sign

        f[nx] = exp_At * est

    return f


# ============================================================
# TESTS
# ============================================================
def test_inversions():
    """
    Test both methods against known Laplace transform pairs.
    """
    print("=" * 70)
    print("TESTING LAPLACE INVERSION METHODS")
    print("=" * 70)

    # Test 1: f(t) = exp(-a*t), F(p) = 1/(p+a)
    a = 2.0
    def g1(p):
        return (1.0 / (p + a)).reshape(-1, 1)

    print("\nTest 1: f(t) = exp(-a*t), a=2")
    for t in [0.5, 1.0, 2.0, 5.0]:
        exact = np.exp(-a * t)
        fs = laplace_invert_stehfest(g1, t)[0]
        fa = laplace_invert_abate_whitt(g1, t)[0]
        print(f"  t={t:.1f}: exact={exact:.8f}, Stehfest={fs:.8f} (err={abs(fs-exact):.2e}), "
              f"AbateWhitt={fa:.8f} (err={abs(fa-exact):.2e})")

    # Test 2: f(t) = t*exp(-a*t), F(p) = 1/(p+a)^2
    def g2(p):
        return (1.0 / (p + a) ** 2).reshape(-1, 1)

    print(f"\nTest 2: f(t) = t*exp(-a*t), a={a}")
    for t in [0.5, 1.0, 2.0, 5.0]:
        exact = t * np.exp(-a * t)
        fs = laplace_invert_stehfest(g2, t)[0]
        fa = laplace_invert_abate_whitt(g2, t)[0]
        print(f"  t={t:.1f}: exact={exact:.8f}, Stehfest={fs:.8f} (err={abs(fs-exact):.2e}), "
              f"AbateWhitt={fa:.8f} (err={abs(fa-exact):.2e})")

    # Test 3: f(t) = sin(w*t), F(p) = w/(p^2+w^2)
    w = 3.0
    def g3(p):
        return (w / (p ** 2 + w ** 2)).reshape(-1, 1)

    print(f"\nTest 3: f(t) = sin(w*t), w={w} (oscillatory)")
    for t in [0.5, 1.0, 2.0, 5.0]:
        exact = np.sin(w * t)
        fs = laplace_invert_stehfest(g3, t)[0]
        fa = laplace_invert_abate_whitt(g3, t)[0]
        print(f"  t={t:.1f}: exact={exact:+.8f}, Stehfest={fs:+.8f} (err={abs(fs-exact):.2e}), "
              f"AbateWhitt={fa:+.8f} (err={abs(fa-exact):.2e})")

    # Test 4: f(t) = 1 (step function), F(p) = 1/p
    def g4(p):
        return (1.0 / p).reshape(-1, 1)

    print(f"\nTest 4: f(t) = 1 (constant)")
    for t in [0.5, 1.0, 2.0, 5.0]:
        exact = 1.0
        fs = laplace_invert_stehfest(g4, t)[0]
        fa = laplace_invert_abate_whitt(g4, t)[0]
        print(f"  t={t:.1f}: exact={exact:.8f}, Stehfest={fs:.8f} (err={abs(fs-exact):.2e}), "
              f"AbateWhitt={fa:.8f} (err={abs(fa-exact):.2e})")

    # Test 5: Multi-output — f(t) = [exp(-t), exp(-2t), exp(-3t)]
    def g5(p):
        # p is array of shape (n_p,)
        # Return shape (n_p, 3)
        result = np.zeros((len(p), 3), dtype=complex if np.iscomplexobj(p) else float)
        for j, a_j in enumerate([1.0, 2.0, 3.0]):
            result[:, j] = 1.0 / (p + a_j)
        return result

    print(f"\nTest 5: Multi-output f(t) = [exp(-t), exp(-2t), exp(-3t)]")
    t = 1.0
    exact = np.array([np.exp(-1), np.exp(-2), np.exp(-3)])
    fs = laplace_invert_stehfest(g5, t)
    fa = laplace_invert_abate_whitt(g5, t)
    for j in range(3):
        print(f"  f_{j+1}: exact={exact[j]:.8f}, Stehfest={fs[j]:.8f} (err={abs(fs[j]-exact[j]):.2e}), "
              f"AbateWhitt={fa[j]:.8f} (err={abs(fa[j]-exact[j]):.2e})")

    # Test 6: Survival probability proxy — F(p) = exp(-sqrt(2p)) / p
    # f(t) = erfc(1/sqrt(2t))
    from scipy.special import erfc
    def g6(p):
        return (np.exp(-np.sqrt(2 * p)) / p).reshape(-1, 1)

    print(f"\nTest 6: f(t) = erfc(1/sqrt(2t)) — survival-type function")
    for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
        exact_val = erfc(1.0 / np.sqrt(2 * t))
        fs = laplace_invert_stehfest(g6, t)[0]
        fa = laplace_invert_abate_whitt(g6, t)[0]
        print(f"  t={t:5.1f}: exact={exact_val:.8f}, Stehfest={fs:.8f} (err={abs(fs-exact_val):.2e}), "
              f"AbateWhitt={fa:.8f} (err={abs(fa-exact_val):.2e})")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_inversions()
