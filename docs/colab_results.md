# First micromagnetic runs on Google Colab (2026-09-04)

All runs used mumax⁺ 1.2.1 through the Colab CLI (see README, "Workflow C").
Cell size 10 × 10 × 5 nm throughout (the paper's 5 nm lateral cells make the
2×2 supercell cost ~50 min per input even on an A100, see "Cost" below).
Readout: ridge regression on 80 MHz bins of the 2–14 GHz power spectrum
(150 outputs; "per-region" = 16 × 150), sequential train/test split, 5-step
washout, ridge parameter chosen on a validation slice of the training data.
"Raw input" = the same regression on (u, u², u³) of the current input.

## Hysteresis calibration (field along x + 1°)

| system | top-layer switching | bottom-layer switching |
|---|---|---|
| 1 unit cell (2 islands) | 32 mT | 35 mT (partially reversed intermediate 32–35 mT) |
| 2×2 supercell (8 islands, 2 % size disorder) | 31 and 35 mT | 54–56 mT; y-island top layers show partial (vortex-like) transitions at +4 / −17 mT |

![hysteresis](figures/hysteresis_2x2_10nm.png)

## Reservoir runs

| run | GPU | system | protocol | task | steps | s/step | reservoir NRMSE (global / per-region) | raw-input NRMSE |
|---|---|---|---|---|---|---|---|---|
| s2sq | T4 | 1×1 | symmetric ±B loop, measured at +B, 28–36 mT | sine → square | 90 | 125 | 0.001 / 0.008 | 0.34 |
| mg10 | A100 | 2×2 | symmetric ±B loop, measured at +B, 28–38 mT | Mackey-Glass t+10 | 80 | 135 | 0.88 / 0.74 | 0.78 |
| mg10alt | A100 | 2×2 | alternating polarity, measured at −1.2 mT | Mackey-Glass t+10 | 80 | 124 | 0.97 / 3.3 | 0.54 |

![s2sq](figures/t4_sine_to_square.png)
![mg10](figures/a100_mg10_bipolar.png)
![mg10alt](figures/a100_mg10_alternating.png)

## What the data say

1. **The spectral fingerprint is an excellent nonlinear basis.** The sine →
   square transformation (which needs a sign function) is learned to 1e-6 MSE
   from the spectrum measured at the loop field; the cubic raw-input baseline
   cannot do it.
2. **Symmetric loops erase memory.** On the 1×1 system the feature vectors at
   equal loop amplitude on rising and falling branches are identical to three
   decimals: a −B → +B loop ending at +B puts every switchable layer in the
   same state whatever happened before. The 2×2 system shows ~7 % history
   dependence (vortex-like states in the y-islands), but no usable past-input
   recall: reconstructing u(t−k) from the spectra is worse than from the
   current input for every k. Any "memory capacity" measured with a periodic
   or quasi-periodic input (sine, Mackey-Glass) is contaminated by the
   signal's own periodicity and must not be quoted.
3. **Alternating polarity as implemented does not help.** Each x-island top
   layer then stores the sign of the last input above its threshold, which
   for a smooth input is essentially the *parity* of the last threshold
   crossing, not an amplitude. The microstates are discrete (28 distinct
   fingerprints in 80 steps), the readout cannot generalise from 38 training
   points, and the polarity leaks into the prediction as a period-2 zigzag.
4. **Where the experimental memory comes from.** In Gartside et al. the
   memory is the progressive, stochastic evolution of the vortex population
   over many field cycles in an array of ~10⁶ islands with a spread of
   coercive fields, read out as smoothly varying mode amplitudes. A 0 K
   micromagnetic supercell of 8 perfect stadiums is deterministic and
   discrete; averaging over many disorder realisations, adding thermal
   fluctuations (`temperature`), edge roughness, and using 200–500 training
   points are the levers to make the simulation behave like the experiment.

## Suggested next experiments

* **Unipolar loops with a partial reset:** apply +B_k (unipolar), measure at
  bias, and after every input apply a fixed −B_leak that lies between the
  softest and hardest switching fields. Elements then record whether recent
  inputs exceeded their threshold since the last leak reset, giving a fading
  memory of amplitudes rather than of polarity.
* **Larger switching-field spread:** `disorder_sigma` 0.05–0.10, or an
  explicit width/length gradient, so that the input range 28–38 mT spans many
  distinct thresholds instead of two.
* **Thermal fluctuations and many seeds:** run several disorder seeds and sum
  their spectra (the flip-chip CPW measures ~10⁶ islands), and/or set a finite
  temperature in the driver so vortex nucleation becomes probabilistic.
* **More data:** ≥ 200 training points; on an A100 at 10 nm cells that is
  ~7 h for the 2×2 supercell with the adaptive stepping.

## Cost (measured)

| grid | cells | GPU | minimisation | 13 ns FMR |
|---|---|---|---|---|
| 80×80×17 (1×1, 10 nm) | 0.11 M | T4 | 0.6 s | 64 s |
| 80×80×17 | 0.11 M | A100 | 0.37 s | 33 s |
| 160×160×17 (2×2, 10 nm) | 0.44 M | A100 | 5.3 s | 82 s |
| 320×320×17 (2×2, 5 nm) | 1.74 M | A100 | 20 s | 420 s |

Minimisation count per input: 64 (symmetric ±34 mT loop, 2 mT steps),
42 with adaptive stepping (6 mT below 26 mT), ~20 for unipolar/alternating.
