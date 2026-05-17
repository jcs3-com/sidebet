# sidebet

A visual calculator that treats sports bets as binary options. NBA moneylines
calibrated as Brownian motion with drift; closed-form Greeks; live in-browser
Monte Carlo. One self-contained HTML file. No build step.

**[live demo →](https://jcs3-com.github.io/sidebet/)**

![preview](preview.png)

---

## what it does

The page is in two parts.

**Module 01 — the trade ticket.** Input American odds for both sides of an
NBA moneyline, a stake, the score differential, and time elapsed. The page
de-vigs the line, calibrates a Brownian-motion-with-drift model to the
implied probabilities, and outputs the five Greeks (Δ, Γ, Θ, 𝒱, and live WP)
plus a payoff surface chart and a theta sheet. It's an options chain for a
basketball game.

**Module 02 — the convexity profile.** Pre-computed (40k Monte Carlo paths
per price point) plus on-demand live MC (5k paths in ~100 ms). Shows the
relationship between pregame underdog probability and expected peak in-game
repricing. The user's current bet is plotted on each curve as an amber
marker so they can see where their position lives on the convexity spectrum.

## the empirical claim

> For NBA moneylines under the model (σ = 13.5), the expected peak in-game
> underdog probability follows a tight log-linear law in entry probability:
>
>     E[max p_dog] / p₀  ≈  1.05 − 0.91 · ln(p₀)        R² = 0.9998
>
> across p₀ ∈ [0.03, 0.49], from 40,000 Monte Carlo paths per price.
>
> The 2× hit rate is approximately constant at 46–50% regardless of p₀.
> Cheap underdogs do not double up *more often* — they double up by larger
> *amounts*. The folk wisdom that "below 25¢ is the line" has the shape
> right and the threshold wrong; there is no kink in the curve.

## the math

A moneyline bet is a binary option on score differential `D(t)`, where
`dD = μ dt + σ dW`. Calibrate `μ` from the de-vigged pregame implied
probability; fix `σ ≈ 13.5` from Stern (1994), replicated by Polson &
Stern (2015). Live favorite WP is

    P_fav(D, t) = Φ((D + μ(1−t)) / (σ√(1−t)))

All four Greeks have closed forms (see `verify_greeks.py` for the SymPy
proof). The standardized argument is `z(D,t) = (D + μ(1−t)) / (σ√(1−t))`
and the Greeks reduce to:

    Δ = φ(z) / (σ√(1−t))
    Γ = −z · φ(z) / (σ² (1−t))
    Θ = φ(z) · (D − μ(1−t)) / (2σ (1−t)^{3/2})
    𝒱 = −φ(z) · z / σ

## caveats (stated on the site too)

1. Constant within-game σ. Real basketball has stochastic vol from foul
   rate, pace shifts, and end-game intentional fouling.
2. Brownian motion assumes continuous scoring. Fits NBA well; **does not**
   fit NFL (discrete possessions). v1 is NBA only.
3. σ = 13.5 is a multi-season average; pace eras shift it ±10–15%. The
   slider lets users see the sensitivity.
4. Model prices ≠ market prices. This is a visualization tool, not an
   edge-finder, not financial advice.

## running it

**Just open `index.html` in a browser.** That's it. The page loads React,
Recharts, and Babel from CDN. No build step.

If you want to verify the math:

```bash
python verify_greeks.py    # SymPy: symbolic confirmation of all 4 Greeks
python verify_fast.py      # numpy: numerical FD checks + empirical claim
python bake_curves.py      # regenerate the pre-computed Module 02 data
```

If you want to run the TypeScript test suite:

```bash
npm install
npm test
```

## files

| file | purpose |
| --- | --- |
| `index.html` | the whole site, self-contained |
| `verify_greeks.py` | SymPy proof that closed-form Greeks match symbolic derivatives |
| `verify_fast.py` | numerical tests + Module 02 empirical sweep |
| `bake_curves.py` | offline computation of the pre-baked convexity curves |
| `src/lib/model.ts` | reference TypeScript port of the model |
| `src/lib/__tests__/model.test.ts` | Vitest suite |
| `convexity_headline.png` | the four-panel summary chart |

## license

MIT. Use it, fork it, rip out the math for your own thing.

## the punch line, one more time

Most sports-bet visualizations are scoreboards. This one is an options
chain. The bet hasn't changed; the lens has.
