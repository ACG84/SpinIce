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

## Escape-field run: the optimum is a feasibility boundary

`--design width t_spacer t_top offset_y --objective escape --steps 10` (four
parameters, 21 min).  The objective is the reordering field of the lowest
level, i.e. the field at which the antiparallel ground state first becomes
degenerate with an excited level: the direct handle on the sink seen in the
transition tables.

| step | width (nm) | spacer (nm) | t_top (nm) | offset (nm) | levels +/+, +/V+, +/V-, V+/-, V+/+ (aJ) | B_reorder (mT) | escape (mT) |
|---|---|---|---|---|---|---|---|
| 0 | 140 | 35.0 | 20.0 | 50.0 | 27.4, 25.2, 29.9, -, 36.7 | 8.3, 12.8, 15.1, -, 23.9 | 12.8 |
| 2 | 149.7 | 35.2 | 15.0 | 46.9 | 23.2, 18.3, 21.7, 25.4, 28.9 | 6.7, 7.5, 9.0, 16.6, 17.3 | 7.5 |
| 4 | 152.0 | 35.3 | 13.1 | 47.3 | 20.8, 15.8, 19.0, 23.4, 26.5 | 5.9, 6.1, 7.4, 15.0, 15.5 | 6.1 |
| 5-10 | 152-155 | 35.3-35.8 | 12.8 -> 6.6 | 47-48 | +/V+ then +/V- lost | | 3.0 (ill-defined) |

* The gradient is dominated by the top-layer thickness: thinning the soft
  layer lowers every level and every reordering field.  Step 4 is the last
  design in which all five addressable levels exist: 152 nm wide, 13 nm top
  layer, escape field halved (12.8 -> 6.1 mT), the vortex levels clustered
  within 16-27 aJ instead of 25-37 aJ, and the mean reordering field down from
  15.0 to 10.0 mT.
* Beyond that the top-layer vortex states cease to exist (a 13 nm, 150 nm wide
  NiFe layer no longer holds a vortex): the loop rejected and halved those
  steps three times each, then (in this run) accepted the loss and kept
  thinning to 6.6 nm, where the escape field reads 3 mT but the level
  structure that makes the ASVI interesting is gone.  The script now stops at
  the boundary and reports the last feasible design instead.
* This is the design tension in one number: the ground state reorders more
  easily with a thin top layer, the extra (vortex) levels need a thick, wide
  one.  A design goal that keeps the vortex levels alive must be a constrained
  optimum on that boundary; the missing constraint is the switching barrier of
  the layers, which the string-method module (`asvi_rc/barriers.py`) provides.

## Switching barriers with the string method (first results, not yet converged)

`asvi_rc/barriers.py` implements the simplified string method (Barzilai-Borwein
descent per image, equal-arc-length re-parametrisation) plus a climbing image,
on the same magnum.np state, so the saddle is a stationary point and the
envelope theorem gives the design gradient of a barrier for free.
`scripts/barrier_demo.py` runs it for the top-layer reversal AP -> P of the
nominal island (10 nm cells, 16 images, ~2-3 min per barrier on 4 cores).

| applied field (mT) | barrier AP -> P (aJ) | level dE(P-AP) (aJ) | path |
|---|---|---|---|
| 0 | 37-39 (string maximum), 31 first hump | 27.4 | two humps: vortex nucleation in the top layer (~31 aJ), shoulder at the +/V+ level (~26 aJ), second hump before P |
| 15 | ~16 | -5.8 | one hump |
| 25 | ~14 | -28.1 | one hump |

Design gradient of the zero-field barrier (autograd at the highest image vs
central finite difference): d/d width 0.293 vs 0.309 aJ/nm; d/d t_spacer
-0.33 aJ/nm; d/d t_top +1.27 aJ/nm (a thicker top layer is harder to switch,
a thicker spacer easier, as expected).

Caveats, to be fixed before barriers enter an objective:

* The string is not converged to the aJ level: the energy of individual images
  still jumps by several aJ between iterations because linear interpolation
  between vortex textures creates unphysical intermediates on the 10 nm grid.
* The climbing image, started from the highest string image, slid down the
  second hump to 0.5 aJ above the P state with a residual torque of 1.5e3 A/m
  (relaxed minima reach 1e-2 A/m), i.e. it did not lock onto a saddle.
* Remedies in order of cost: split the path at the vortex intermediate
  (AP -> +/V+ and +/V+ -> P separately), more images with geodesic (slerp)
  re-parametrisation, climb only once the string tangent is stable, 5 nm
  cells (4x cost).  The 15 and 25 mT barriers already show the expected
  collapse toward the ~30 mT switching field of the top layer.

The physical picture is useful even now: the top layer reverses through
vortex nucleation, so the switching barrier and the +/V+ level are the same
physics, which is why widening the island lowered both in the reorder run.

## Memory-curve proxy: a differentiable soft automaton (`asvi_rc/softautomaton.py`)

The GPU transition tables showed the reservoir is a finite automaton on the
relaxed states driven by the loop amplitude, and that its memory is set by
which transitions the protocol can reach.  The proxy makes that automaton
smooth and differentiable using only quantities the design model provides:

* states i with energies E_i and axis moments M_i (from the relaxations, with
  design gradients by the envelope theorem);
* single-layer transitions i -> j with probability
  p_ij(B) = sigmoid((dM B - dE - barrier) / (|dM| width)), i.e. the Zeeman gain
  of the switch must exceed an effective barrier (70 aJ, calibrated so the
  nominal top layer switches at 29.7 mT vs 30 mT measured) with a 2 mT
  switching width; a few cascade rounds per field stage;
* the state-probability vector rho_t propagated through the field stages of
  each protocol step (leak excursion, write excursion, bias), a ridge readout
  of u(t-k) from rho_t, test R^2(k), and MC = sum_k R^2(k) (smoothly clamped).

Everything is a torch expression, so dMC/d(design) and dMC/d(protocol) come
from one backward pass.  On the nominal island it reproduces the GPU findings
qualitatively: with the original protocol (28-38 mT, leak 33+-5, bias -20) the
automaton visits ~1.4 bits of its 16 states and recalls only the current input
(MC 0.56, R^2(1) ~ 0).  `tests/test_softautomaton.py` covers it.

### Protocol optimisation on a fixed landscape (`scripts/protocol_design.py`, ~1 s per step)

| protocol | free parameters | start MC | optimised MC | R^2(k = 0..3) | optimised fields (mT) |
|---|---|---|---|---|---|
| leak | b_min, b_max, leak, jitter, bias | 0.56 | 2.08 | 0.95, 0.90, 0.06, 0.00 | window 25.2-29.9, leak 33.4 +- 1.2, bias -26.8 |
| unipolar | b_min, b_max, bias | 0.55 | 1.20 | 0.53, 0.24, 0.12, 0.08 | window 35.7-36.8, bias -18.3 |

* The leak protocol turns into a one-step memory: the write window straddles
  the top-layer switching field so u_t decides whether the layer switched,
  and the deepened bias makes the reset conditional on the previous state
  (occupancy 2.25 bits).  The unipolar protocol finds a slower-decaying curve
  instead.  Two different memory shapes from the same 16-state landscape,
  found in 30 gradient steps each.
* Caveats: one island driven along its own axis (the lattice and the 45 deg
  axis are the next step, same code with the unit-cell catalogue); the barrier
  is a single calibrated constant until the string-method barriers converge;
  the readout uses state probabilities, whereas the real readout is the FMR
  spectrum, which is a further (state -> spectrum) map that the GPU tables
  showed to be injective enough on the visited states.

### Joint design with the memory objective (`--objective memory`, 16 tracked states)

`--design width t_spacer t_top --objective memory --steps 8` with the
optimised leak protocol (29 min, 16 relaxations per step, all 16 states alive
after the seeding fix below):

| step | width (nm) | spacer (nm) | t_top (nm) | MC | R^2(k = 0..2) |
|---|---|---|---|---|---|
| 0 | 140 | 35.0 | 20.0 | 2.07 | 0.98, 0.88, 0.06 |
| 1 | 145.6 | 35.3 | 17.5 | 2.05 | 0.97, 0.86, 0.05 |
| 2 | 142.6 | 35.1 | 18.8 | 2.09 | 0.98, 0.88, 0.05 |
| 8 | 141.7 | 34.0 | 18.6 | 2.09 | 0.99, 0.88, 0.06 |

* For this protocol the nominal geometry sits on a flat optimum of the proxy:
  eight steps move the design by a few nm and MC by 0.02 (the step decay
  prevents the oscillation seen in the first run).  Optimising the protocol
  (0.56 -> 2.08) mattered far more than the geometry (2.07 -> 2.09).
* The ceiling is structural: one island driven along its own axis has 16
  states, of which the protocol can use about 2.2 bits, giving one step of
  memory plus the current input.  More memory needs more reachable states,
  i.e. the lattice unit cell (256 states, 45 deg drive of both sublattices).
  The proxy, the protocol optimiser and the design loop take the unit cell
  unchanged (energies and moments from its catalogue), which is the next run.
* Seeding note: in the first run only 9-10 of the 16 seeded states survived.
  Cause: the soft mask puts 12.5 % of Ms into the cell just outside every
  interface, those cells carried no seed and were filled with the island axis
  direction, so a uniform "cap" sat on every seeded vortex and steered the
  relaxation (2 states lost on the nominal grid, 7 with the taller grid of the
  design loop).  The driver now dilates the seed texture into the fractional
  cells; all 16 states then relax as seeded in every setup.

### Correction: the readout sees one state, not a distribution

The first version of the proxy fed the state *probabilities* rho_t to the
ridge readout.  With a 2 mT switching width that makes rho_t a smooth
function of the input, and the readout "recalled" information that a one-shot
measurement of a single magnetic state never contains: the MC of 2.08 and the
R^2(1) = 0.90 above are artefacts of that.  The memory curve is now the
expected R^2 of a readout on the *sampled* state (conditional means given the
state, computed exactly from rho_t; `softautomaton.memory_curve`), which a
sampled stochastic automaton reproduces (`asvi_rc/force.py`, lag-0 recall
R^2 0.55 sampled vs 0.64 expected).  With the corrected proxy the single
island has no memory at all: re-optimising the leak protocol gives at best
R^2(0) = 0.64 (a two-threshold encoding of the current input with a 5-55 mT
window and readout at zero field) and R^2(k >= 1) = 0, because the leak
excursion erases the previous write.  This agrees with the GPU runs
(Mackey-Glass NRMSE 1.0, no past-input recall) and with the physical
argument: one island driven along its axis is a one- or two-bit latch, not a
reservoir.  The earlier protocol and design tables above are superseded by
`docs/data/protocol_leak_v2*`.

## FORCE learning (`asvi_rc/force.py`, `scripts/force_rc.py`)

Online recursive-least-squares readout with the output fed back into the next
loop amplitude, B_{t+1} = b_min + (b_max - b_min) clip(u_{t+1} + g z_t, 0, 1)
(a controller between the FMR readout and the field coil).  Reservoirs: the
soft automaton sampled as a stochastic automaton, or a GPU transition table
(deterministic, amplitudes snapped to the table grid).  Tasks: delayed recall,
NARMA-2, autonomous sine generation.

| reservoir | protocol | lag-0 recall R^2 | lag-1 | NARMA-2 | sine | feedback gain effect |
|---|---|---|---|---|---|---|
| single island, 16 states (soft, sampled) | 5-55 mT, leak 33.8, bias 0 | 0.55 | 0.00 | 0.00 | 0.00 | none (g = 0.3, 0.6, 1.0) |
| unit cell, GPU 45 deg table (11 states, 8 mT grid) | 31-47 / 40-56 mT, leak 39-48 +- 8, bias -20 | 0.00 | 0.00 | 0.00 | 0.00 | none: stuck in one state |

* Feedback cannot create memory where the open loop has none: on the single
  island the fed-back output only modulates a one-bit write that the leak
  erases; on the GPU table the leak drives the cell into the +/-/+/+ sink
  (a fixed point of every excursion in the table) and nothing moves again.
* FORCE therefore needs what the whole analysis keeps pointing at: an
  automaton with a mutually reachable core of many states inside the
  protocol window.  The unit-cell landscape (253 states) under a 45 deg drive
  is the first candidate; the sampled soft automaton runs it in seconds once
  its protocol is optimised (in progress).

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
  for the energies); barrier gradients need the string convergence work
  listed above before they can constrain a design run.
