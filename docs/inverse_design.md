# magnum.np backend and gradient-based inverse design (Sep 5, CPU)

Colab stopped handing out GPUs (free T4 quota exhausted, A100 out of compute
units), so the catalogue / energy-landscape part of the proxy pipeline now also
runs on [magnum.np](https://gitlab.com/magnum.np/magnum.np) (PyTorch, CPU or
GPU) through `asvi_rc/magnumnp_driver.py`, and its autodiff turns the
"cheap proxy" idea into real gradient-based design.  Everything below ran on
the 4-core CPU of this container.

## Validation against mumax+ (single island, 10 nm cells, 64 x 32 x 17)

`python mumaxplus/state_catalogue.py --unit single --cell-xy 10e-9 --fast --backend magnumnp --out runs/np_single_10`
(171 s for all 16 seeds, 3-10 s per relaxation after the first torch.compile):

| state | mumax+ (aJ) | magnum.np (aJ) |
|---|---|---|
| +/- , -/+ | 0.00 | 0.000 |
| +/V+ | 26.45 | 26.446 |
| +/+ | 27.57 | 27.568 |
| +/V- | 30.68 | 30.677 |
| V+/- | 32.78 | 32.775 |
| V+/+ | 36.75 | 36.752 |
| V+/V- | 44.09 | 44.095 |
| V+/V+ | 54.72 | 54.720 |

All 16 seeds stable, every level within 0.01 aJ; the absolute energies agree
too (37.324 aJ for the antiparallel state in both codes).  The two codes share
the same finite-difference discretisation, so this is the expected agreement.

## Differentiable geometry

Design parameters: island length and width, the three layer thicknesses, the
top-layer offset, Ms of each layer and A.  Every cell's Ms and A is a C1
spline mask of these (cells within one cell of a boundary carry a fractional
occupation, all others are exactly 0 or 1).  Relaxed states are stationary in
m, so dE_state/d(design) = dE/d(design) at fixed m (envelope theorem): one
backward pass per state, no differentiation through the minimiser.

Two implementation details mattered:

* magnum.np's exchange field divides by Ms; that is NaN-safe only forward, so
  the driver evaluates the exchange energy division-free (harmonic-mean bond
  stiffness) when gradients are needed.
* `torch.clamp` has zero gradient at its bounds, which silently halved the
  width and Ms sensitivities whenever an island edge sat exactly on a cell
  boundary (140 nm = 14 cells).  The mask ramps are therefore C1 quadratic
  splines.

Check (splitting dE = E(+/+) - E(+/-) at 10 nm, soft mask 26.6 aJ vs hard mask 27.6 aJ):

| sensitivity | autograd | central finite difference |
|---|---|---|
| d(dE)/d width (J/m) | 2.849e-10 | 2.850e-10 |
| d(dE)/d t_spacer (J/m) | -9.146e-10 | -9.096e-10 |
| d(dE)/d Ms_top (J m/A) | 3.313e-23 | 3.313e-23 |
| dE_AP/d width (J/m) | 2.072e-10 | 2.074e-10 |

i.e. widening the island by 1 nm raises the parallel-state cost by 0.28 aJ,
thickening the spacer by 1 nm lowers it by 0.91 aJ (the hard-mask secant from
the mumax+ spacer scan is 0.28-0.36 aJ/nm over 10-15 nm; the local derivative
of the one-cell-smoothed island is steeper).

`tests/test_magnumnp.py` (coarse 20 nm island, 28 s) checks the hard/soft
agreement and the gradient against finite differences.

## Inverse design loop (`scripts/inverse_design.py`)

Tracks the two antiparallel ground states and the five field-addressable
levels (+/+, +/V+, +/V-, V+/-, V+/+) through relaxations as the design moves,
builds the objective from their energies and axis moments as a torch
expression, and takes normalised gradient steps with bounds.  Objectives:
`spread` (standard deviation of the levels), `reorder` (mean reordering field
B_i = dE_i / max_g |M_i - M_g|, the definition of `catalogue_analysis.py`),
`escape` (the reordering field of the lowest level).  A penalty keeps every
level at least `--min-level` (default 5 aJ) above the ground state so the
optimiser cannot make a vortex the new ground state.  Cost: one relaxation
per tracked state per step, ~40-60 s per step at 10 nm on 4 cores.

DEMO_TABLE

## What to do with it

* The `escape` objective is the direct handle on the sink found in the
  transition tables (the antiparallel ground state never reorders inside the
  protocol window): it is the reordering field of the lowest level.
* Multi-parameter runs (`--design width t_spacer t_top offset_y`) cost the
  same per step; the gradient tells which knob matters.
* The same driver runs the periodic unit cell (`--unit cell`, 10 nm: 80 x 80 x
  17, ~15 s per relaxation), so interlayer-plus-lattice objectives are
  feasible on the CPU; transition tables (hundreds of relaxations) still want
  a GPU.
* Limitations: the moment gradients neglect the response of m* (exact only
  for the energies); barriers (switching fields) are not differentiated yet -
  a string-method / NEB energy barrier with the same envelope trick is the
  natural extension, and magnum.np ships a `StringSolver`.
