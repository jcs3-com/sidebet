/**
 * Side-Bet / Convex: Core model module
 *
 * Models a moneyline sports bet as a binary option on a Brownian motion
 * with drift, where the underlying is score differential D(t).
 *
 * Model:
 *   dD(t) = mu * dt + sigma * dW(t)
 *   t in [0, 1], normalized game time (0 = tipoff, 1 = final buzzer)
 *   D(t) = current score differential (favorite minus underdog)
 *   mu   = drift per unit time = expected final margin from pregame spread
 *   sigma = volatility per unit time (points per sqrt(game-fraction))
 *           For NBA, sigma ~= 13.5 (Stern 1994, refined by Polson-Stern 2015)
 *
 * Live win probability for the favorite:
 *   P_fav(D, t) = Phi(z(D, t))
 *   z(D, t) = (D + mu*(1-t)) / (sigma * sqrt(1-t))
 *
 * All Greeks are derived in closed form and verified symbolically.
 * See /verify_greeks.py for the SymPy verification.
 *
 * REFERENCES:
 * - Stern, H. (1994). A Brownian motion model for the progress of sports scores.
 *   JASA 89(427), 1128-1134.
 * - Polson, N. & Stern, H. (2015). The implied volatility of a sports game.
 *   Journal of Quantitative Analysis in Sports.
 */

// ---------------------------------------------------------------------------
// Normal distribution primitives
// ---------------------------------------------------------------------------

const SQRT_2 = Math.SQRT2;
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const INV_SQRT_2PI = 1 / SQRT_2PI;

/**
 * Standard normal PDF.
 */
export function phi(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

/**
 * Standard normal CDF using Abramowitz & Stegun 7.1.26 approximation
 * to erf. Accurate to ~1.5e-7 over the whole real line, which is well
 * beyond what we need for a probability display.
 */
export function Phi(x: number): number {
  return 0.5 * (1 + erf(x / SQRT_2));
}

/**
 * Inverse standard normal CDF (probit function).
 * Beasley-Springer-Moro algorithm. Accurate to ~1e-9.
 */
export function PhiInv(p: number): number {
  if (p <= 0 || p >= 1) {
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    throw new Error(`PhiInv: p must be in (0, 1), got ${p}`);
  }

  // Constants (Beasley-Springer-Moro)
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number, r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}

/**
 * Error function via Abramowitz & Stegun 7.1.26.
 * Max error ~1.5e-7.
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) *
      t +
      0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return sign * y;
}

// ---------------------------------------------------------------------------
// Model parameters and calibration
// ---------------------------------------------------------------------------

/**
 * Sport-specific volatility constants for score differential over the full game.
 * Units: points per sqrt(full-game).
 *
 * NBA: ~13.5 (Stern 1994, Polson-Stern 2015, replicated across multiple sources)
 * Other sports are placeholders pending empirical calibration. Do NOT ship
 * other sports without recalibrating.
 */
export const SPORT_SIGMA: Record<string, number> = {
  NBA: 13.5,
  // NFL: discrete possessions, Brownian model is wrong shape -- needs different model
  // MLB: 4.0 (rough), low-scoring discrete events
  // NHL: 2.5 (rough), very low-scoring
};

export interface ModelParams {
  /** Drift per unit time (expected final margin in points, favorite minus underdog) */
  mu: number;
  /** Volatility per unit time (points per sqrt(game-fraction)) */
  sigma: number;
}

/**
 * Convert American moneyline odds to implied probability (with vig).
 *
 * +150 -> 100/250 = 0.40
 * -200 -> 200/300 = 0.667
 */
export function americanToImpliedProb(odds: number): number {
  if (odds === 0) throw new Error("Odds cannot be zero");
  if (odds > 0) {
    return 100 / (odds + 100);
  } else {
    return -odds / (-odds + 100);
  }
}

/**
 * Convert implied probability back to American odds.
 */
export function impliedProbToAmerican(p: number): number {
  if (p <= 0 || p >= 1) throw new Error(`p must be in (0,1), got ${p}`);
  if (p >= 0.5) {
    return -Math.round((p / (1 - p)) * 100);
  } else {
    return Math.round(((1 - p) / p) * 100);
  }
}

/**
 * De-vig two-sided moneyline odds using proportional (multiplicative) method.
 *
 * Given favorite and underdog American odds, returns the de-vigged
 * implied probability for the favorite.
 *
 * Example: -190 / +155 -> raw probs 0.655 and 0.392, sum = 1.047 (4.7% vig)
 * De-vigged favorite prob = 0.655 / 1.047 = 0.626
 */
export function devigMoneyline(favOdds: number, dogOdds: number): {
  favProb: number;
  dogProb: number;
  vig: number;
} {
  const rawFav = americanToImpliedProb(favOdds);
  const rawDog = americanToImpliedProb(dogOdds);
  const total = rawFav + rawDog;
  return {
    favProb: rawFav / total,
    dogProb: rawDog / total,
    vig: total - 1,
  };
}

/**
 * Calibrate model from pregame implied probability and assumed sigma.
 *
 * Given p0 = P(favorite wins) pregame, and sigma:
 *   p0 = Phi(mu / sigma)  (since at t=0, z = mu/sigma)
 *   => mu = sigma * PhiInv(p0)
 */
export function calibrateFromPregameProb(
  p0: number,
  sigma: number,
): ModelParams {
  if (p0 <= 0 || p0 >= 1) {
    throw new Error(`Pregame probability must be in (0,1), got ${p0}`);
  }
  const mu = sigma * PhiInv(p0);
  return { mu, sigma };
}

/**
 * Alternative calibration: from pregame spread.
 * If favorite is laid -X.5 points, then mu = X.5.
 * Sigma defaults to sport constant.
 */
export function calibrateFromSpread(
  spread: number,
  sigma: number = SPORT_SIGMA.NBA,
): ModelParams {
  return { mu: spread, sigma };
}

// ---------------------------------------------------------------------------
// Core: standardized argument
// ---------------------------------------------------------------------------

/**
 * Standardized z-score for the live WP formula.
 *
 * z(D, t) = (D + mu*(1-t)) / (sigma * sqrt(1-t))
 *
 * Note: at t=1, this blows up. The model formally only applies for t < 1.
 * In practice, at t=1, the game is over and P is 0 or 1 by terminal state.
 */
export function zScore(D: number, t: number, params: ModelParams): number {
  if (t >= 1) {
    // At/after game end, return signed infinity based on outcome
    if (D > 0) return Infinity;
    if (D < 0) return -Infinity;
    return 0;
  }
  if (t < 0) throw new Error(`t must be in [0, 1), got ${t}`);
  const { mu, sigma } = params;
  return (D + mu * (1 - t)) / (sigma * Math.sqrt(1 - t));
}

// ---------------------------------------------------------------------------
// Win probabilities
// ---------------------------------------------------------------------------

/**
 * Live win probability for the FAVORITE given state (D, t).
 */
export function winProbabilityFav(
  D: number,
  t: number,
  params: ModelParams,
): number {
  if (t >= 1) {
    return D > 0 ? 1 : D < 0 ? 0 : 0.5;
  }
  return Phi(zScore(D, t, params));
}

/**
 * Live win probability for the UNDERDOG.
 */
export function winProbabilityDog(
  D: number,
  t: number,
  params: ModelParams,
): number {
  return 1 - winProbabilityFav(D, t, params);
}

// ---------------------------------------------------------------------------
// Greeks
//
// All derivatives are with respect to the FAVORITE's win probability.
// For the underdog, negate sign on delta/gamma/theta and flip vega
// (since dog_wp = 1 - fav_wp).
// ---------------------------------------------------------------------------

export interface Greeks {
  /** dP/dD: sensitivity of WP to 1-point change in score differential */
  delta: number;
  /** d2P/dD2: convexity in score differential */
  gamma: number;
  /** dP/dt: sensitivity to game-time elapsed (per unit of normalized time) */
  theta: number;
  /** dP/dsigma: sensitivity to game volatility (per 1.0 increase in sigma) */
  vega: number;
}

/**
 * Compute all four Greeks at state (D, t) under params (mu, sigma).
 * Greeks are for the FAVORITE's WP. Negate for the underdog.
 *
 * For interpretation in bet-value terms, multiply by the bet's payout structure.
 * For a $1 stake on the favorite at decimal odds d:
 *   bet_value(D, t) = P_fav(D, t) * d - 1
 *   => bet_delta = d * delta, etc.
 */
export function computeGreeks(
  D: number,
  t: number,
  params: ModelParams,
): Greeks {
  if (t >= 1) {
    // Game over: all sensitivities collapse to zero
    return { delta: 0, gamma: 0, theta: 0, vega: 0 };
  }
  if (t < 0) throw new Error(`t must be in [0, 1), got ${t}`);

  const { mu, sigma } = params;
  const remaining = 1 - t;
  const sqrtRem = Math.sqrt(remaining);
  const z = (D + mu * remaining) / (sigma * sqrtRem);
  const phiZ = phi(z);

  // delta = phi(z) / (sigma * sqrt(1-t))
  const delta = phiZ / (sigma * sqrtRem);

  // gamma = -z * phi(z) / (sigma^2 * (1-t))
  const gamma = (-z * phiZ) / (sigma * sigma * remaining);

  // theta = phi(z) * dz/dt
  // dz/dt = (D - mu*(1-t)) / (2 * sigma * (1-t)^(3/2))
  const dz_dt = (D - mu * remaining) / (2 * sigma * Math.pow(remaining, 1.5));
  const theta = phiZ * dz_dt;

  // vega = -phi(z) * z / sigma
  const vega = (-phiZ * z) / sigma;

  return { delta, gamma, theta, vega };
}

/**
 * Greeks for the UNDERDOG's WP.
 * dP_dog/dX = -dP_fav/dX for all X (since P_dog = 1 - P_fav).
 */
export function computeGreeksDog(
  D: number,
  t: number,
  params: ModelParams,
): Greeks {
  const g = computeGreeks(D, t, params);
  return {
    delta: -g.delta,
    gamma: -g.gamma,
    theta: -g.theta,
    vega: -g.vega,
  };
}

/**
 * Convexity ratio: |gamma| / delta.
 * Higher = more option-like (more convexity per unit of directional exposure).
 * Lower  = more bet-like (mostly directional).
 *
 * This is the headline metric for Module 2's claim.
 */
export function convexityRatio(
  D: number,
  t: number,
  params: ModelParams,
): number {
  const g = computeGreeks(D, t, params);
  if (g.delta === 0) return Infinity;
  return Math.abs(g.gamma) / g.delta;
}

// ---------------------------------------------------------------------------
// Path simulation (for Module 2)
// ---------------------------------------------------------------------------

/**
 * Box-Muller transform for standard normal sampling.
 * Returns one N(0,1) variate. (For perf, could memoize the second.)
 */
export function randn(rng: () => number = Math.random): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Mulberry32 seeded RNG for reproducible simulations.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface PathResult {
  /** Score differential at each step (length M+1, includes t=0) */
  D: number[];
  /** Favorite WP at each step */
  pFav: number[];
  /** Underdog WP at each step */
  pDog: number[];
  /** Time grid */
  t: number[];
  /** Final outcome: true if favorite won */
  favWon: boolean;
}

/**
 * Simulate one game path under the model.
 *
 * @param params (mu, sigma)
 * @param steps  number of time steps (default 200 ~ 14s per step in NBA)
 * @param rng    RNG function (default Math.random); pass mulberry32(seed) for reproducibility
 */
export function simulatePath(
  params: ModelParams,
  steps: number = 200,
  rng: () => number = Math.random,
): PathResult {
  const { mu, sigma } = params;
  const dt = 1 / steps;
  const sqrtDt = Math.sqrt(dt);

  const D = new Array<number>(steps + 1);
  const pFav = new Array<number>(steps + 1);
  const pDog = new Array<number>(steps + 1);
  const t = new Array<number>(steps + 1);

  D[0] = 0;
  t[0] = 0;
  pFav[0] = Phi(mu / sigma);
  pDog[0] = 1 - pFav[0];

  for (let i = 1; i <= steps; i++) {
    const dW = randn(rng) * sqrtDt;
    D[i] = D[i - 1] + mu * dt + sigma * dW;
    t[i] = i * dt;
    if (i === steps) {
      // Snap to terminal
      pFav[i] = D[i] > 0 ? 1 : D[i] < 0 ? 0 : 0.5;
    } else {
      pFav[i] = Phi(zScore(D[i], t[i], params));
    }
    pDog[i] = 1 - pFav[i];
  }

  return {
    D,
    pFav,
    pDog,
    t,
    favWon: D[steps] > 0,
  };
}

// ---------------------------------------------------------------------------
// Module 2 aggregates: convexity profile
// ---------------------------------------------------------------------------

export interface ConvexityStats {
  /** Pregame underdog probability used */
  p0: number;
  /** Mean of max underdog WP achieved during games */
  meanMaxPDog: number;
  /** Median of max underdog WP */
  medianMaxPDog: number;
  /** Mean of min underdog WP */
  meanMinPDog: number;
  /** P(max P_dog >= 2 * p0): the convexity hit rate */
  doubleUpRate: number;
  /** Expected peak multiple: E[max P_dog] / p0 */
  expectedPeakMultiple: number;
  /** Fraction of paths where the underdog won */
  dogWinRate: number;
  /** Convexity ratio at entry */
  entryConvexityRatio: number;
  /** Number of paths simulated */
  nPaths: number;
}

/**
 * Run a convexity profile for one pregame underdog probability p0.
 *
 * Calibrates model so that pregame favorite WP = 1 - p0, then simulates
 * nPaths games and aggregates path statistics.
 */
export function convexityProfile(
  p0: number,
  options: {
    sigma?: number;
    nPaths?: number;
    steps?: number;
    seed?: number;
  } = {},
): ConvexityStats {
  const sigma = options.sigma ?? SPORT_SIGMA.NBA;
  const nPaths = options.nPaths ?? 10000;
  const steps = options.steps ?? 200;
  const seed = options.seed;

  const pFav = 1 - p0;
  const params = calibrateFromPregameProb(pFav, sigma);
  const rng = seed !== undefined ? mulberry32(seed) : Math.random;

  const maxDogs: number[] = new Array(nPaths);
  const minDogs: number[] = new Array(nPaths);
  let dogWins = 0;

  for (let i = 0; i < nPaths; i++) {
    const path = simulatePath(params, steps, rng);
    let maxD = path.pDog[0];
    let minD = path.pDog[0];
    for (let j = 1; j < path.pDog.length; j++) {
      if (path.pDog[j] > maxD) maxD = path.pDog[j];
      if (path.pDog[j] < minD) minD = path.pDog[j];
    }
    maxDogs[i] = maxD;
    minDogs[i] = minD;
    if (!path.favWon) dogWins++;
  }

  const meanMaxPDog = maxDogs.reduce((a, b) => a + b, 0) / nPaths;
  const meanMinPDog = minDogs.reduce((a, b) => a + b, 0) / nPaths;
  const sortedMax = [...maxDogs].sort((a, b) => a - b);
  const medianMaxPDog = sortedMax[Math.floor(nPaths / 2)];
  const doubleUpRate = maxDogs.filter((m) => m >= 2 * p0).length / nPaths;

  const entryConvexityRatio = convexityRatio(0, 0, params);

  return {
    p0,
    meanMaxPDog,
    medianMaxPDog,
    meanMinPDog,
    doubleUpRate,
    expectedPeakMultiple: meanMaxPDog / p0,
    dogWinRate: dogWins / nPaths,
    entryConvexityRatio,
    nPaths,
  };
}

/**
 * Sweep convexity profiles across a range of underdog prices.
 * Returns an array of stats, one per price point.
 */
export function convexityProfileSweep(
  prices: number[],
  options: { sigma?: number; nPaths?: number; steps?: number; seed?: number } = {},
): ConvexityStats[] {
  return prices.map((p0, i) =>
    convexityProfile(p0, {
      ...options,
      seed: options.seed !== undefined ? options.seed + i : undefined,
    }),
  );
}
