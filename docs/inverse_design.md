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

Demo: `--design width t_spacer --objective reorder --steps 8` (10 nm cells, 683 s):

| step | width (nm) | spacer (nm) | levels +/+, +/V+, +/V-, V+/-, V+/+ (aJ) | B_reorder (mT) | mean B (mT) |
|---|---|---|---|---|---|
| 0 | 140 | 35.0 | 27.4, 25.2, 29.9, -, 36.7 | 8.3, 12.8, 15.1, -, 23.9 | 15.1 |
| 2 | 160 | 35.6 | 33.1, 23.3, 27.8, 25.6, 29.7 | 9.1, 10.5, 12.6, 17.3, 18.4 | 13.6 |
| 4 | 180 | 36.3 | 38.7, 21.1, 25.4, 19.4, 23.5 | 9.7, 8.8, 10.7, 11.7, 13.2 | 10.9 |
| 6 | 200 | 37.3 | 44.0, 19.2, 23.4, 13.6, 17.5 | 10.3, 7.6, 9.1, 7.5, 9.1 | 8.7 |
| 8 | 220 | 38.5 | 48.7, 17.8, 21.1, 8.0, 11.2 | 10.8, 6.5, 7.9, 4.0, 5.7 | 7.0 |

* Step 0 reproduces the mumax+ catalogue analysis of the nominal island
  (8.4 / 12.8 / 15.2 / 22.0 / 24.8 mT at 5 nm cells), so the soft model and the
  reordering-field definition are consistent with the GPU pipeline.  (V+/- did
  not survive its seed at step 0 in the soft model and joins from step 1.)
* The gradient is dominated by the width (spacer barely moves in normalised
  units): widening the island lowers every vortex level and their reordering
  fields (V+/+ 23.9 -> 5.7 mT), while the parallel macrospin state gets more
  expensive (27 -> 49 aJ) but its reordering field only creeps up (8.3 -> 10.8
  mT) because its moment difference grows with the width too.
* The loop would keep widening until the vortex levels hit the 5 aJ floor
  (V+/- is at 8 aJ at 220 nm): the "optimum" of this objective is an island on
  the verge of preferring vortices, which is exactly where the level structure
  is most clustered and the reordering fields smallest, but also where the
  macrospin bits stop being robust.  The width bound / level floor are the
  real design constraints; a switching-field (barrier) term is the missing
  ingredient (see below).

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
