"""
Symbolic verification of the Greeks for the Brownian-bridge WP model.

P_fav(D, t) = Phi( (D + mu*(1-t)) / (sigma * sqrt(1-t)) )

We compute partial derivatives symbolically and confirm closed-form matches.
"""
import sympy as sp

D, t, mu, sigma = sp.symbols('D t mu sigma', real=True, positive=False)
sigma_pos = sp.Symbol('sigma', real=True, positive=True)

# Standardized argument
z = (D + mu*(1 - t)) / (sigma_pos * sp.sqrt(1 - t))

# Normal CDF and PDF (using sympy's erf representation)
def Phi(x):
    return sp.Rational(1, 2) * (1 + sp.erf(x / sp.sqrt(2)))

def phi(x):
    return sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi)

P_fav = Phi(z)

print("=" * 70)
print("MODEL: P_fav(D, t) = Phi((D + mu*(1-t)) / (sigma*sqrt(1-t)))")
print("=" * 70)

# --- DELTA: dP/dD ---
delta_symbolic = sp.simplify(sp.diff(P_fav, D))
delta_closed = phi(z) / (sigma_pos * sp.sqrt(1 - t))
delta_diff = sp.simplify(delta_symbolic - delta_closed)
print(f"\nDELTA check: symbolic - closed_form = {delta_diff}")
print(f"  Closed form: phi(z) / (sigma * sqrt(1-t))")

# --- GAMMA: d2P/dD2 ---
gamma_symbolic = sp.simplify(sp.diff(P_fav, D, 2))
gamma_closed = -z * phi(z) / (sigma_pos**2 * (1 - t))
gamma_diff = sp.simplify(gamma_symbolic - gamma_closed)
print(f"\nGAMMA check: symbolic - closed_form = {gamma_diff}")
print(f"  Closed form: -z * phi(z) / (sigma^2 * (1-t))")

# --- THETA: dP/dt ---
# This is the one I want to verify carefully.
theta_symbolic = sp.simplify(sp.diff(P_fav, t))
# My proposed closed form:
dz_dt = sp.diff(z, t)
theta_proposed = phi(z) * dz_dt
theta_proposed_simplified = sp.simplify(theta_proposed)
theta_diff = sp.simplify(theta_symbolic - theta_proposed_simplified)
print(f"\nTHETA check: symbolic - phi(z)*dz/dt = {theta_diff}")

# Let's express dz/dt cleanly
dz_dt_simplified = sp.simplify(dz_dt)
print(f"\n  dz/dt simplified: {dz_dt_simplified}")

# Even cleaner form
dz_dt_explicit = sp.together(dz_dt_simplified)
print(f"  dz/dt as single fraction: {dz_dt_explicit}")

# --- VEGA: dP/dsigma ---
vega_symbolic = sp.simplify(sp.diff(P_fav, sigma_pos))
vega_closed = -phi(z) * z / sigma_pos
vega_diff = sp.simplify(vega_symbolic - vega_closed)
print(f"\nVEGA check: symbolic - closed_form = {vega_diff}")
print(f"  Closed form: -phi(z) * z / sigma")

# --- Sanity checks at specific values ---
print("\n" + "=" * 70)
print("NUMERICAL SANITY CHECKS")
print("=" * 70)

# Case 1: Pregame (t=0), tied expected outcome (mu=0)
# Should give P_fav = 0.5
val = P_fav.subs([(D, 0), (t, 0), (mu, 0), (sigma_pos, 13.5)])
print(f"\nP_fav(D=0, t=0, mu=0, sigma=13.5) = {float(val):.4f}  (expect 0.5)")

# Case 2: Favorite up 10 with mu=5, sigma=13.5, t=0.5
val = P_fav.subs([(D, 10), (t, 0.5), (mu, 5), (sigma_pos, 13.5)])
print(f"P_fav(D=10, t=0.5, mu=5, sigma=13.5) = {float(val):.4f}")

# Case 3: Delta at tied late game (should be very large)
val = delta_closed.subs([(D, 0), (t, 0.95), (mu, 0), (sigma_pos, 13.5)])
print(f"Delta(D=0, t=0.95, mu=0, sigma=13.5) = {float(val):.4f}")

# Case 4: Vega for an underdog (P_fav < 0.5) should be negative for favorite
# meaning positive for underdog
val = vega_closed.subs([(D, 0), (t, 0), (mu, -5), (sigma_pos, 13.5)])
print(f"Vega for favorite when mu=-5 (i.e. underdog team): {float(val):.6f}")
print(f"  (Should be POSITIVE: increasing sigma helps the team that's behind in expectation)")

# Case 5: Gamma sign
# When favorite is BEHIND (D < 0 with mu small), z < 0, so -z > 0, gamma > 0
val = gamma_closed.subs([(D, -5), (t, 0.5), (mu, 0), (sigma_pos, 13.5)])
print(f"Gamma when favorite is behind by 5, mid-game: {float(val):.6f}")
print(f"  (Should be POSITIVE: convexity helps the team behind)")
