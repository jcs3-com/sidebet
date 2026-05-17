/**
 * Unit tests for model.ts
 *
 * Run with: npx vitest run
 *
 * These tests catch the most likely classes of bugs:
 *  1. Sign errors in Greeks
 *  2. Calibration round-trip failures
 *  3. Boundary behavior at t=0 and t->1
 *  4. Numerical equivalence of analytic vs. finite-difference Greeks
 *  5. Monte Carlo convergence to the closed-form pregame WP
 */

import { describe, it, expect } from "vitest";
import {
  Phi,
  PhiInv,
  phi,
  americanToImpliedProb,
  impliedProbToAmerican,
  devigMoneyline,
  calibrateFromPregameProb,
  zScore,
  winProbabilityFav,
  winProbabilityDog,
  computeGreeks,
  computeGreeksDog,
  convexityRatio,
  simulatePath,
  mulberry32,
  convexityProfile,
  SPORT_SIGMA,
  ModelParams,
} from "../model";

const TOL = 1e-6;
const LOOSE_TOL = 1e-3;

// ---------------------------------------------------------------------------
// Normal distribution primitives
// ---------------------------------------------------------------------------

describe("Phi (standard normal CDF)", () => {
  it("Phi(0) = 0.5", () => {
    expect(Phi(0)).toBeCloseTo(0.5, 6);
  });
  it("Phi(-inf) = 0, Phi(inf) = 1", () => {
    expect(Phi(-10)).toBeCloseTo(0, 6);
    expect(Phi(10)).toBeCloseTo(1, 6);
  });
  it("Phi(1.96) ~ 0.975", () => {
    expect(Phi(1.96)).toBeCloseTo(0.975, 3);
  });
  it("symmetry: Phi(x) + Phi(-x) = 1", () => {
    for (const x of [0.5, 1.0, 1.5, 2.0, 3.0]) {
      expect(Phi(x) + Phi(-x)).toBeCloseTo(1, 6);
    }
  });
});

describe("PhiInv (probit)", () => {
  it("PhiInv(0.5) = 0", () => {
    expect(PhiInv(0.5)).toBeCloseTo(0, 6);
  });
  it("PhiInv(0.975) ~ 1.96", () => {
    expect(PhiInv(0.975)).toBeCloseTo(1.96, 2);
  });
  it("round-trip: Phi(PhiInv(p)) = p", () => {
    for (const p of [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]) {
      expect(Phi(PhiInv(p))).toBeCloseTo(p, 6);
    }
  });
});

describe("phi (standard normal PDF)", () => {
  it("phi(0) = 1/sqrt(2pi)", () => {
    expect(phi(0)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 6);
  });
  it("symmetry: phi(x) = phi(-x)", () => {
    for (const x of [0.5, 1.0, 2.0]) {
      expect(phi(x)).toBeCloseTo(phi(-x), 12);
    }
  });
});

// ---------------------------------------------------------------------------
// Odds conversion
// ---------------------------------------------------------------------------

describe("American odds conversion", () => {
  it("+100 = 0.5", () => {
    expect(americanToImpliedProb(100)).toBeCloseTo(0.5, 6);
  });
  it("-100 = 0.5", () => {
    expect(americanToImpliedProb(-100)).toBeCloseTo(0.5, 6);
  });
  it("+200 = 1/3", () => {
    expect(americanToImpliedProb(200)).toBeCloseTo(1 / 3, 6);
  });
  it("-200 = 2/3", () => {
    expect(americanToImpliedProb(-200)).toBeCloseTo(2 / 3, 6);
  });
  it("round-trip: impliedProbToAmerican(americanToImpliedProb(x)) = x", () => {
    for (const odds of [-300, -200, -150, -110, 110, 150, 200, 300]) {
      const p = americanToImpliedProb(odds);
      expect(impliedProbToAmerican(p)).toBe(odds);
    }
  });
});

describe("devigMoneyline", () => {
  it("removes vig such that probs sum to 1", () => {
    const { favProb, dogProb, vig } = devigMoneyline(-190, 155);
    expect(favProb + dogProb).toBeCloseTo(1, 10);
    expect(vig).toBeGreaterThan(0);
  });
  it("favorite has higher de-vigged prob than underdog", () => {
    const { favProb, dogProb } = devigMoneyline(-190, 155);
    expect(favProb).toBeGreaterThan(dogProb);
  });
  it("symmetric pick'em (-110 / -110) gives ~50/50", () => {
    const { favProb, dogProb } = devigMoneyline(-110, -110);
    expect(favProb).toBeCloseTo(0.5, 6);
    expect(dogProb).toBeCloseTo(0.5, 6);
  });
});

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

describe("calibrateFromPregameProb", () => {
  it("pregame WP from calibrated params equals input p0", () => {
    for (const p0 of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const params = calibrateFromPregameProb(p0, 13.5);
      const wp = winProbabilityFav(0, 0, params);
      expect(wp).toBeCloseTo(p0, 6);
    }
  });
  it("p0=0.5 gives mu=0", () => {
    const params = calibrateFromPregameProb(0.5, 13.5);
    expect(params.mu).toBeCloseTo(0, 6);
  });
  it("higher p0 gives positive mu", () => {
    const params = calibrateFromPregameProb(0.7, 13.5);
    expect(params.mu).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// zScore boundary behavior
// ---------------------------------------------------------------------------

describe("zScore", () => {
  it("at t=0, z = mu/sigma", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    expect(zScore(0, 0, params)).toBeCloseTo(5 / 13.5, 6);
  });
  it("at t=1 with D>0, z = +inf", () => {
    expect(zScore(5, 1, { mu: 0, sigma: 13.5 })).toBe(Infinity);
  });
  it("at t=1 with D<0, z = -inf", () => {
    expect(zScore(-5, 1, { mu: 0, sigma: 13.5 })).toBe(-Infinity);
  });
});

// ---------------------------------------------------------------------------
// Win probability behavior
// ---------------------------------------------------------------------------

describe("winProbability boundary cases", () => {
  it("favorite leading at game end -> WP=1", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    expect(winProbabilityFav(5, 1, params)).toBe(1);
    expect(winProbabilityDog(5, 1, params)).toBe(0);
  });
  it("favorite trailing at game end -> WP=0", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    expect(winProbabilityFav(-5, 1, params)).toBe(0);
    expect(winProbabilityDog(-5, 1, params)).toBe(1);
  });
  it("WP increases monotonically in D, all else equal", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    const t = 0.5;
    let prev = winProbabilityFav(-20, t, params);
    for (let D = -15; D <= 20; D += 5) {
      const cur = winProbabilityFav(D, t, params);
      expect(cur).toBeGreaterThan(prev);
      prev = cur;
    }
  });
  it("pDog + pFav = 1", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    for (const [D, t] of [
      [0, 0],
      [5, 0.3],
      [-3, 0.7],
      [10, 0.9],
    ] as [number, number][]) {
      expect(
        winProbabilityFav(D, t, params) + winProbabilityDog(D, t, params),
      ).toBeCloseTo(1, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// Greeks: analytic vs. finite differences
// ---------------------------------------------------------------------------

describe("Greeks vs. finite differences", () => {
  const params: ModelParams = { mu: 5, sigma: 13.5 };

  const testPoints: [number, number][] = [
    [0, 0.1],
    [3, 0.3],
    [-2, 0.5],
    [8, 0.7],
    [-5, 0.85],
  ];

  it("delta matches finite difference dP/dD", () => {
    const eps = 0.01;
    for (const [D, t] of testPoints) {
      const fd =
        (winProbabilityFav(D + eps, t, params) -
          winProbabilityFav(D - eps, t, params)) /
        (2 * eps);
      const analytic = computeGreeks(D, t, params).delta;
      expect(analytic).toBeCloseTo(fd, 4);
    }
  });

  it("gamma matches second finite difference d2P/dD2", () => {
    const eps = 0.05;
    for (const [D, t] of testPoints) {
      const fd =
        (winProbabilityFav(D + eps, t, params) -
          2 * winProbabilityFav(D, t, params) +
          winProbabilityFav(D - eps, t, params)) /
        (eps * eps);
      const analytic = computeGreeks(D, t, params).gamma;
      expect(analytic).toBeCloseTo(fd, 3);
    }
  });

  it("theta matches finite difference dP/dt", () => {
    const eps = 0.0001;
    for (const [D, t] of testPoints) {
      const fd =
        (winProbabilityFav(D, t + eps, params) -
          winProbabilityFav(D, t - eps, params)) /
        (2 * eps);
      const analytic = computeGreeks(D, t, params).theta;
      expect(analytic).toBeCloseTo(fd, 3);
    }
  });

  it("vega matches finite difference dP/dsigma", () => {
    const eps = 0.01;
    for (const [D, t] of testPoints) {
      const pp: ModelParams = { mu: params.mu, sigma: params.sigma + eps };
      const pm: ModelParams = { mu: params.mu, sigma: params.sigma - eps };
      const fd =
        (winProbabilityFav(D, t, pp) - winProbabilityFav(D, t, pm)) / (2 * eps);
      const analytic = computeGreeks(D, t, params).vega;
      expect(analytic).toBeCloseTo(fd, 4);
    }
  });
});

describe("Greeks: sign and magnitude properties", () => {
  it("delta is always positive for favorite", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    for (const [D, t] of [
      [-10, 0.1],
      [0, 0.5],
      [10, 0.9],
    ] as [number, number][]) {
      expect(computeGreeks(D, t, params).delta).toBeGreaterThan(0);
    }
  });

  it("gamma is positive when favorite is behind expected line", () => {
    // When z < 0, gamma = -z*phi(z)/(sigma^2 * (1-t)) > 0
    // z < 0 means D + mu*(1-t) < 0
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    expect(computeGreeks(-5, 0.5, params).gamma).toBeGreaterThan(0);
  });

  it("gamma is negative when favorite is ahead of expected line", () => {
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    expect(computeGreeks(5, 0.5, params).gamma).toBeLessThan(0);
  });

  it("vega is positive for underdog (negative for favorite when ahead)", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    // Pregame, favorite ahead in expectation -> vega for fav is negative
    expect(computeGreeks(0, 0, params).vega).toBeLessThan(0);
    // And positive for dog
    expect(computeGreeksDog(0, 0, params).vega).toBeGreaterThan(0);
  });

  it("delta peaks at z=0 (when D = -mu*(1-t))", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    const t = 0.5;
    const D_peak = -params.mu * (1 - t); // z = 0 here
    const deltaPeak = computeGreeks(D_peak, t, params).delta;
    const deltaAway = computeGreeks(D_peak + 10, t, params).delta;
    expect(deltaPeak).toBeGreaterThan(deltaAway);
  });

  it("absolute Greeks decrease for blowouts (far from z=0)", () => {
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    const t = 0.5;
    const gClose = computeGreeks(0, t, params);
    const gBlowout = computeGreeks(50, t, params);
    expect(Math.abs(gBlowout.delta)).toBeLessThan(Math.abs(gClose.delta));
    expect(Math.abs(gBlowout.gamma)).toBeLessThan(0.0001);
  });

  it("theta magnitude grows as t -> 1 for close games", () => {
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    const thetaEarly = computeGreeks(0, 0.3, params).theta;
    const thetaLate = computeGreeks(0, 0.9, params).theta;
    expect(Math.abs(thetaLate)).toBeGreaterThan(Math.abs(thetaEarly));
  });
});

describe("computeGreeksDog", () => {
  it("dog Greeks = -fav Greeks", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    const fav = computeGreeks(2, 0.4, params);
    const dog = computeGreeksDog(2, 0.4, params);
    expect(dog.delta).toBeCloseTo(-fav.delta, 10);
    expect(dog.gamma).toBeCloseTo(-fav.gamma, 10);
    expect(dog.theta).toBeCloseTo(-fav.theta, 10);
    expect(dog.vega).toBeCloseTo(-fav.vega, 10);
  });
});

// ---------------------------------------------------------------------------
// Monte Carlo convergence
// ---------------------------------------------------------------------------

describe("simulatePath", () => {
  it("path length matches steps+1", () => {
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    const path = simulatePath(params, 100);
    expect(path.D.length).toBe(101);
    expect(path.pFav.length).toBe(101);
    expect(path.t.length).toBe(101);
  });

  it("starts at D=0, t=0", () => {
    const params: ModelParams = { mu: 5, sigma: 13.5 };
    const path = simulatePath(params, 100);
    expect(path.D[0]).toBe(0);
    expect(path.t[0]).toBe(0);
    expect(path.pFav[0]).toBeCloseTo(Phi(5 / 13.5), 6);
  });

  it("ends at t=1", () => {
    const params: ModelParams = { mu: 0, sigma: 13.5 };
    const path = simulatePath(params, 100);
    expect(path.t[100]).toBeCloseTo(1, 10);
  });

  it("Monte Carlo dog win rate converges to pregame dog prob", () => {
    // With p0=0.20 underdog, ~20% of paths should result in dog wins
    const p0_dog = 0.2;
    const params = calibrateFromPregameProb(1 - p0_dog, SPORT_SIGMA.NBA);
    const rng = mulberry32(42);
    let dogWins = 0;
    const N = 5000;
    for (let i = 0; i < N; i++) {
      const path = simulatePath(params, 100, rng);
      if (!path.favWon) dogWins++;
    }
    const empirical = dogWins / N;
    // Should be within ~2 std errs of 0.20.
    // SE ~ sqrt(0.2 * 0.8 / 5000) ~ 0.0057, so 2 SE ~ 0.011
    expect(empirical).toBeGreaterThan(0.18);
    expect(empirical).toBeLessThan(0.22);
  });
});

describe("convexityProfile (Module 2 core)", () => {
  it("returns dog win rate close to p0 for many paths", () => {
    const stats = convexityProfile(0.15, {
      nPaths: 5000,
      steps: 100,
      seed: 123,
    });
    expect(stats.dogWinRate).toBeGreaterThan(0.13);
    expect(stats.dogWinRate).toBeLessThan(0.17);
  });

  it("meanMaxPDog >= p0 always (max can only increase or stay)", () => {
    for (const p0 of [0.05, 0.15, 0.25, 0.4]) {
      const stats = convexityProfile(p0, { nPaths: 2000, steps: 100, seed: 7 });
      expect(stats.meanMaxPDog).toBeGreaterThanOrEqual(p0);
    }
  });

  it("expected peak multiple decreases as p0 increases (the headline claim)", () => {
    const results = [0.05, 0.1, 0.2, 0.3, 0.4].map((p0) =>
      convexityProfile(p0, { nPaths: 3000, steps: 100, seed: 99 }),
    );
    // Should be monotone decreasing in p0
    for (let i = 1; i < results.length; i++) {
      expect(results[i].expectedPeakMultiple).toBeLessThan(
        results[i - 1].expectedPeakMultiple,
      );
    }
  });

  it("doubleUpRate decreases as p0 increases", () => {
    const results = [0.05, 0.15, 0.3, 0.45].map((p0) =>
      convexityProfile(p0, { nPaths: 3000, steps: 100, seed: 55 }),
    );
    for (let i = 1; i < results.length; i++) {
      expect(results[i].doubleUpRate).toBeLessThanOrEqual(
        results[i - 1].doubleUpRate + 0.02, // allow small MC noise
      );
    }
  });
});

// ---------------------------------------------------------------------------
// convexityRatio
// ---------------------------------------------------------------------------

describe("convexityRatio", () => {
  it("higher for deep underdog (z far from 0) than for pickem", () => {
    const paramsLong = calibrateFromPregameProb(0.9, 13.5); // 10% dog
    const paramsClose = calibrateFromPregameProb(0.55, 13.5); // 45% dog
    const cLong = convexityRatio(0, 0, paramsLong);
    const cClose = convexityRatio(0, 0, paramsClose);
    expect(cLong).toBeGreaterThan(cClose);
  });
});
