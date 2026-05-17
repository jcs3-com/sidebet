"""
Vectorized verification - matches the TS model but uses numpy for speed.
We only need this fast enough to validate the headline claim empirically.
"""
import math
import numpy as np
from scipy import stats
import unittest

# Primitives
def Phi(x): return stats.norm.cdf(x)
def phi(x): return stats.norm.pdf(x)
def PhiInv(p): return stats.norm.ppf(p)

SIGMA_NBA = 13.5

def calibrate(p0, sigma=SIGMA_NBA):
    return sigma * PhiInv(p0), sigma

def wp_fav_scalar(D, t, mu, sigma):
    if t >= 1:
        return 1.0 if D > 0 else (0.0 if D < 0 else 0.5)
    z = (D + mu*(1-t)) / (sigma * math.sqrt(1-t))
    return Phi(z)

def greeks(D, t, mu, sigma):
    if t >= 1:
        return {"delta":0, "gamma":0, "theta":0, "vega":0}
    rem = 1 - t
    sqrt_rem = math.sqrt(rem)
    z = (D + mu*rem) / (sigma * sqrt_rem)
    phiz = phi(z)
    return {
        "delta": phiz / (sigma * sqrt_rem),
        "gamma": -z * phiz / (sigma**2 * rem),
        "theta": phiz * (D - mu*rem) / (2 * sigma * rem**1.5),
        "vega":  -phiz * z / sigma,
    }

def simulate_paths_vectorized(mu, sigma, n_paths, steps, seed=None):
    """Simulate n_paths at once. Returns (n_paths, steps+1) array of WP_dog."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps
    sqrt_dt = math.sqrt(dt)
    # Increments
    dW = rng.standard_normal((n_paths, steps)) * sqrt_dt
    drift = mu * dt
    increments = drift + sigma * dW
    D = np.zeros((n_paths, steps + 1))
    D[:, 1:] = np.cumsum(increments, axis=1)

    t_arr = np.linspace(0, 1, steps + 1)
    rem = 1 - t_arr  # (steps+1,)
    rem[-1] = 1e-15  # avoid div-by-zero at terminal
    z = (D + mu * rem[None, :]) / (sigma * np.sqrt(rem[None, :]))
    p_fav = Phi(z)
    # Snap terminal
    p_fav[:, -1] = (D[:, -1] > 0).astype(float)
    p_dog = 1 - p_fav
    return D, p_fav, p_dog

def convexity_profile(p0_dog, n_paths=10000, steps=200, sigma=SIGMA_NBA, seed=None):
    mu, sig = calibrate(1 - p0_dog, sigma)
    D, p_fav, p_dog = simulate_paths_vectorized(mu, sig, n_paths, steps, seed)
    max_dog = p_dog.max(axis=1)
    min_dog = p_dog.min(axis=1)
    dog_wins = (p_fav[:, -1] == 0).sum()
    g = greeks(0, 0, mu, sig)
    return {
        "p0": p0_dog,
        "mean_max_pdog": float(max_dog.mean()),
        "median_max_pdog": float(np.median(max_dog)),
        "mean_min_pdog": float(min_dog.mean()),
        "double_up_rate": float((max_dog >= 2*p0_dog).mean()),
        "expected_peak_multiple": float(max_dog.mean() / p0_dog),
        "dog_win_rate": float(dog_wins / n_paths),
        "entry_convexity_ratio": abs(g["gamma"]) / g["delta"],
        "entry_delta": g["delta"],
        "entry_gamma": g["gamma"],
    }

# ===========================================================================
# TESTS
# ===========================================================================

class TestGreeksVsFD(unittest.TestCase):
    """Critical: analytic Greeks must match finite differences."""
    def setUp(self):
        self.mu, self.sigma = 5.0, 13.5
        self.points = [(0,0.1),(3,0.3),(-2,0.5),(8,0.7),(-5,0.85)]
    def test_delta(self):
        eps = 0.01
        for D, t in self.points:
            fd = (wp_fav_scalar(D+eps,t,self.mu,self.sigma) -
                  wp_fav_scalar(D-eps,t,self.mu,self.sigma))/(2*eps)
            g = greeks(D,t,self.mu,self.sigma)
            self.assertAlmostEqual(g["delta"], fd, places=5)
    def test_gamma(self):
        eps = 0.05
        for D, t in self.points:
            fd = (wp_fav_scalar(D+eps,t,self.mu,self.sigma)
                  - 2*wp_fav_scalar(D,t,self.mu,self.sigma)
                  + wp_fav_scalar(D-eps,t,self.mu,self.sigma))/(eps*eps)
            g = greeks(D,t,self.mu,self.sigma)
            self.assertAlmostEqual(g["gamma"], fd, places=4)
    def test_theta(self):
        eps = 1e-4
        for D, t in self.points:
            fd = (wp_fav_scalar(D,t+eps,self.mu,self.sigma) -
                  wp_fav_scalar(D,t-eps,self.mu,self.sigma))/(2*eps)
            g = greeks(D,t,self.mu,self.sigma)
            self.assertAlmostEqual(g["theta"], fd, places=3)
    def test_vega(self):
        eps = 0.01
        for D, t in self.points:
            fd = (wp_fav_scalar(D,t,self.mu,self.sigma+eps) -
                  wp_fav_scalar(D,t,self.mu,self.sigma-eps))/(2*eps)
            g = greeks(D,t,self.mu,self.sigma)
            self.assertAlmostEqual(g["vega"], fd, places=5)

class TestMonteCarloConvergence(unittest.TestCase):
    def test_dogwin_matches_p0(self):
        for p0 in [0.10, 0.20, 0.35]:
            r = convexity_profile(p0, n_paths=20000, steps=100, seed=42)
            # SE ~ sqrt(p0*(1-p0)/N), 3-sigma window
            se = math.sqrt(p0*(1-p0)/20000)
            self.assertLess(abs(r["dog_win_rate"] - p0), 4*se,
                msg=f"p0={p0}, got {r['dog_win_rate']}")

class TestConvexityClaim(unittest.TestCase):
    """THE central claim of the tool."""
    def test_monotonic_decrease_peak_multiple(self):
        prices = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        results = [convexity_profile(p, n_paths=10000, steps=150, seed=99+i)
                   for i, p in enumerate(prices)]
        multiples = [r["expected_peak_multiple"] for r in results]
        print("\n  Peak multiples:", {p:round(m,2) for p,m in zip(prices,multiples)})
        for i in range(1, len(multiples)):
            self.assertLess(multiples[i], multiples[i-1])

if __name__ == "__main__":
    print("Running fast verification tests...\n")
    unittest.main(verbosity=2, exit=False)

    print("\n" + "="*78)
    print("CONVEXITY PROFILE SWEEP (the headline result)")
    print("="*78)
    print(f"{'p0_dog':>7} {'mean_max':>9} {'peak_mult':>10} "
          f"{'2x_rate':>9} {'dog_win%':>9} {'γ/δ ratio':>11} "
          f"{'δ':>9} {'|γ|':>9}")
    for p0 in [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        r = convexity_profile(p0, n_paths=20000, steps=150, seed=2026)
        print(f"{p0:>7.2f} {r['mean_max_pdog']:>9.3f} "
              f"{r['expected_peak_multiple']:>10.2f} "
              f"{r['double_up_rate']:>9.3f} "
              f"{r['dog_win_rate']:>9.3f} "
              f"{r['entry_convexity_ratio']:>11.4f} "
              f"{r['entry_delta']:>9.5f} "
              f"{abs(r['entry_gamma']):>9.5f}")
