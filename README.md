# sidebet

A visual calculator that treats sports bets as binary options. NBA moneylines
calibrated as Brownian motion with drift; closed-form Greeks; live in-browser
Monte Carlo. One self-contained HTML file. No build step. No React. No fuss.

**[live demo →](https://YOURUSERNAME.github.io/sidebet/)**

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
> *amounts*.

## the math

A moneyline bet is a binary option on score differential `D(t)`, where
`dD = μ dt + σ dW`. Calibrate `μ` from the de-vigged pregame implied
probability; fix `σ ≈ 13.5` from Stern (1994), replicated by Polson & Stern
(2015). Live favorite WP is

    P_fav(D, t) = Φ((D + μ(1−t)) / (σ√(1−t)))

The Greeks have closed-form expressions:

    Δ = φ(z) / (σ√(1−t))
    Γ = −z · φ(z) / (σ² (1−t))
    Θ = φ(z) · (D − μ(1−t)) / (2σ (1−t)^{3/2})
    𝒱 = −φ(z) · z / σ

where `z(D,t) = (D + μ(1−t)) / (σ√(1−t))`.

## tech stack

Single HTML file. Vanilla JavaScript for the calculator. Chart.js for the
charts (one CDN script, no peer dependencies). All math inlined. All
pre-computed Monte Carlo data inlined.

If Chart.js fails to load (CDN outage), the calculator still works — only
the charts get hidden and a red banner explains what happened.

## running it

**Just open `index.html` in a browser.** That's it.

## caveats (stated plainly on the site)

1. Constant within-game σ. Real basketball has stochastic vol from foul
   rate, pace shifts, and end-game intentional fouling.
2. Brownian motion assumes continuous scoring. Fits NBA well; does not
   fit NFL (discrete possessions). v1 is NBA only.
3. σ = 13.5 is a multi-season average; pace eras shift it ±10–15%.
4. Model prices ≠ market prices. This is a visualization tool, not an
   edge-finder, not financial advice.

## license

MIT.
