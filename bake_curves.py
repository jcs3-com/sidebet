"""
Pre-compute the convexity profile curves at high resolution.
Output JS-ready array literal that gets pasted into index.html.

We use 40k paths per price point and 24 price points for high-quality curves.
This runs once offline; the result is baked into the page.
"""
import math
import numpy as np
from scipy import stats
import json

SIGMA_NBA = 13.5

def Phi(x): return stats.norm.cdf(x)
def phi(x): return stats.norm.pdf(x)
def PhiInv(p): return stats.norm.ppf(p)

def simulate_paths_vec(mu, sigma, n, steps, seed):
    rng = np.random.default_rng(seed)
    dt = 1.0/steps; sqrt_dt = math.sqrt(dt)
    dW = rng.standard_normal((n, steps)) * sqrt_dt
    D = np.zeros((n, steps+1))
    D[:, 1:] = np.cumsum(mu*dt + sigma*dW, axis=1)
    t = np.linspace(0, 1, steps+1)
    rem = 1 - t; rem[-1] = 1e-15
    z = (D + mu*rem[None,:]) / (sigma*np.sqrt(rem[None,:]))
    p_fav = Phi(z)
    p_fav[:,-1] = (D[:,-1] > 0).astype(float)
    return D, 1 - p_fav

# Sweep many p0 values
prices = [round(0.03 + i*0.02, 3) for i in range(24)]  # 0.03, 0.05, ..., 0.49
results = []
print("Computing convexity sweep (40k paths each)...")
for i, p0 in enumerate(prices):
    mu = SIGMA_NBA * PhiInv(1 - p0)
    _, pdog = simulate_paths_vec(mu, SIGMA_NBA, 40000, 200, seed=2026 + i)
    max_d = pdog.max(axis=1)
    z0 = mu / SIGMA_NBA
    delta0 = phi(z0) / SIGMA_NBA
    gamma0 = -z0 * phi(z0) / SIGMA_NBA**2
    row = {
        "p0": p0,
        "meanMax": round(float(max_d.mean()), 5),
        "p10": round(float(np.quantile(max_d, 0.10)), 5),
        "p25": round(float(np.quantile(max_d, 0.25)), 5),
        "p50": round(float(np.quantile(max_d, 0.50)), 5),
        "p75": round(float(np.quantile(max_d, 0.75)), 5),
        "p90": round(float(np.quantile(max_d, 0.90)), 5),
        "peakMult": round(float(max_d.mean() / p0), 4),
        "doubleUp": round(float((max_d >= 2*p0).mean()), 4),
        "tripleUp": round(float((max_d >= 3*p0).mean()), 4),
        "gammaDelta": round(abs(gamma0) / delta0, 5),
    }
    results.append(row)
    print(f"  p0={p0:.2f}  peakMult={row['peakMult']:.2f}  2x={row['doubleUp']:.2f}  γ/δ={row['gammaDelta']:.4f}")

# Compact JSON for inlining
js = "const CONVEXITY_CURVE = " + json.dumps(results, separators=(',', ':')) + ";"
print(f"\nGenerated JS: {len(js)} chars")
open('/home/claude/sidebet/convexity_data.js', 'w').write(js)

# Also fit the log-linear model for display
ln_p = np.log([r["p0"] for r in results])
peak = np.array([r["peakMult"] for r in results])
slope, intercept = np.polyfit(ln_p, peak, 1)
ss_res = np.sum((peak - (intercept + slope*ln_p))**2)
ss_tot = np.sum((peak - peak.mean())**2)
r2 = 1 - ss_res/ss_tot
print(f"\nFit: peakMult = {intercept:.4f} + ({slope:.4f}) * ln(p0)")
print(f"R² = {r2:.6f}")
print(f"\nSo: peakMult ≈ {intercept:.2f} − {abs(slope):.2f}·ln(p₀)")
