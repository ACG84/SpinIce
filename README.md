# SpinIce – reservoir computing with a 3D multilayered artificial spin-vortex ice

Simulation tooling for using the **3D multilayered artificial spin-vortex ice (ASVI)** of
Dion *et al.*, *Nat. Commun.* **15**, 4077 (2024)
([doi:10.1038/s41467-024-48080-z](https://doi.org/10.1038/s41467-024-48080-z))
as a **physical reservoir**, driven with the spin-wave-fingerprint protocol of
Gartside *et al.*, *Nat. Nanotechnol.* **17**, 460 (2022)
([arXiv:2107.08941](https://arxiv.org/abs/2107.08941)).
Both **mumax3** and **mumax⁺** front-ends are provided; they share the same
geometry builder, input-encoding, spectral analysis and readout code.

```
asvi_rc/                shared numpy code (no GPU needed)
  params.py             ASVIParams (geometry/material/discretisation), ProtocolParams (RC protocol)
  geometry.py           periodic supercell, stadium islands, regions, macrospin/vortex initial states
  tasks.py              sine/saw/Mackey-Glass/NARMA datasets, input -> minor-field-loop schedule
  spectra.py            m(t) -> FMR power spectra -> feature vectors; mumax3 table parser
  readout.py            ridge regression, train/test, MSE/NRMSE, memory capacity, raw-input baseline
  mumax3_writer.py      emits unrolled .mx3 scripts (reservoir protocol, field sweep)
  mumaxplus_driver.py   ASVISimulation: the same protocol on the mumax+ Python API
  mock_reservoir.py     CPU toy (macrospins + dipoles) to test the pipeline without a GPU
mumaxplus/run_reservoir.py    GPU: collect reservoir spectra with mumax+
mumaxplus/run_field_sweep.py  GPU: FMR-vs-field map + hysteresis (input-range calibration)
scripts/generate_mumax3.py    write .mx3 scripts for mumax3 (reservoir / sweep)
scripts/process_mumax3_table.py   mumax3 table.txt -> spectra.npz
scripts/train_readout.py      train/evaluate the linear readout on any spectra.npz
scripts/demo_mock.py          end-to-end demo on the toy reservoir (seconds, CPU)
tests/                        pytest suite (includes a fake mumax+ module for a dry run)
```

## The physical system (defaults in `ASVIParams`)

| quantity | value | source |
|---|---|---|
| island footprint | 550 nm × 140 nm, rounded ends | Dion 2024 |
| stack (bottom→top) | NiFe 30 nm / Al 35 nm / NiFe 20 nm | Dion 2024 |
| top-layer lateral offset | 50 nm along ŷ (breaks vortex chirality symmetry) | Dion 2024 |
| lattice | square ASI, 125 nm island-end→vertex gap ⇒ a = 800 nm | Dion 2024 |
| Ms, Aex, α | 800 kA/m, 13 pJ/m, 0.001 | Dion 2024 Methods |
| cells | 5 × 5 × 5 nm (paper: 4.198 × 4.198 × 10 nm) | commensurate with 30/35/20 nm |
| FMR excitation | sinc along ẑ, f_cut = 15 GHz, 1 mT, 26 ns, sampled every 33 ps | Dion 2024 Methods |
| boundary conditions | periodic in-plane supercell of `n_cells` unit cells (2 islands each) | Dion 2024 Methods |
| quenched disorder | 2 % relative std of island length/width per 3D island | needed for a spread of switching fields |

Each magnetic layer of each 3D island can be a ± macrospin or a CW/ACW vortex
(16 states per island). The strong inter-layer dipolar coupling gives a
parallel→antiparallel transition at 5–10 mT and antiparallel→parallel at
26–30 mT (field along x̂), a ~1 GHz zero-field shift between those states and
an acoustic/optical mode gap of 6.5 GHz. Those state-dependent spectra are the
reservoir's readout.

The supercell is one `Ferromagnet` / one mumax3 grid: both NiFe layers plus the
empty Al spacer cells, so inter-layer coupling is the ordinary demagnetising
field and no exchange crosses the spacer. Every layer of every island is a
**region** (id `1 + 2*island + layer`); the table / output contains the
region-averaged magnetisation, i.e. the spin-wave fingerprint of every element.

## The reservoir protocol (defaults in `ProtocolParams`)

1. Saturate along −x̂ (200 mT).
2. For every input value `u_k ∈ [0,1]`: map linearly to a loop amplitude
   `B_k ∈ [b_min, b_max]` (default 20–30 mT), apply a quasi-static **bipolar minor loop**
   `→ −B_k → +B_k` along x̂ (+1°) in 1 mT increments (energy minimisation at every increment),
   then stop at the **measurement field** (`+B_k`, or a small bias field with `measure_at="bias"`).
3. Excite with the broadband sinc pulse, record the region-averaged `m(t)` for 26 ns,
   subtract the static state, FFT → power spectrum. The binned spectrum
   (2–14 GHz in 40 MHz bins = 300 outputs; ×16 if `--per-region`) is the reservoir output of step *k*.
4. Offline: ridge regression on the outputs of the *current* step only
   (no time multiplexing – all memory must be physical), sequential train/test split,
   MSE / NRMSE, plus a "raw input" baseline regression to show what the reservoir adds.

Tasks (`asvi_rc.tasks.TASKS`): `sine_to_square`, `sine_to_saw`, `sine_to_nonlinear`,
`isaw_to_sine`, `isaw_to_square`, `isaw_to_nonlinear`, `mackey_glass_1`,
`mackey_glass_10`, `narma10`.

**Calibrate the input range first.** The 20–30 mT default straddles the
antiparallel→parallel switching window of the experimental sample; with your
own cell size, disorder and supercell the switching fields will differ.
Run a field sweep, look at the per-region hysteresis, and set `b_min`/`b_max`
to bracket the window where only part of the array switches.

## Installation

```bash
pip install -e .            # numpy / scipy / matplotlib; nothing GPU-related
python -m pytest -q         # 16 tests, ~30 s, CPU only
python scripts/demo_mock.py --task sine_to_square   # toy end-to-end run -> runs/mock_demo/demo.png
```

Micromagnetics needs an NVIDIA GPU and one of

* **mumax3** – binary release from <https://mumax.github.io/download.html>
* **mumax⁺** – Python API, <https://github.com/mumax/plus> (pre-built wheels on the
  GitHub releases page, `conda install hcc::mumaxplus`, or build from source with
  `git clone --recursive`, `conda env create -f environment.yml`, `pip install .`).

## Workflow A – mumax⁺

```bash
# 1. calibration: hysteresis + FMR-vs-field map (Dion Fig. 2 style)
python mumaxplus/run_field_sweep.py --b-start=-30e-3 --b-stop 60e-3 --b-step 1e-3 \
    --angle 1 --n-cells 2 2 --out runs/sweep
# 2. reservoir data collection (resumable with --resume)
python mumaxplus/run_reservoir.py --task mackey_glass_10 --n 400 --n-cells 2 2 \
    --b-min 20e-3 --b-max 30e-3 --out runs/mg10
# 3. readout
python scripts/train_readout.py runs/mg10/spectra.npz --n-train 200 --plot runs/mg10/readout.png
python scripts/train_readout.py runs/mg10/spectra.npz --per-region --memory
```

`ASVISimulation` (in `asvi_rc/mumaxplus_driver.py`) can also be used interactively:

```python
from asvi_rc import ASVIParams, ProtocolParams, tasks
from asvi_rc.mumaxplus_driver import ASVISimulation
p, proto = ASVIParams(n_cells=(2, 2)), ProtocolParams()
sim = ASVISimulation(p); sim.saturate(proto)
u, y = tasks.make_task("narma10", 300)
for step in tasks.field_schedule(u, proto):
    sim.apply_ramp(step["ramp"], proto)
    freqs, power = sim.fmr(tasks.field_vector(step["b_meas"], proto))   # (nf,), (nf, 16)
```

## Workflow B – mumax3

```bash
python scripts/generate_mumax3.py sweep --b-start=-30e-3 --b-stop 60e-3 --out mumax3/out/sweep
mumax3 mumax3/out/sweep/asvi_field_sweep.mx3
python scripts/process_mumax3_table.py mumax3/out/sweep/asvi_field_sweep.out/table.txt \
       mumax3/out/sweep/asvi_field_sweep.meta.json      # -> spectra.npz with per-field m_static

python scripts/generate_mumax3.py reservoir --task mackey_glass_10 --n 400 --out mumax3/out/mg10
mumax3 mumax3/out/mg10/asvi_reservoir.mx3
python scripts/process_mumax3_table.py mumax3/out/mg10/asvi_reservoir.out/table.txt \
       mumax3/out/mg10/asvi_reservoir.meta.json
python scripts/train_readout.py mumax3/out/mg10/spectra.npz --n-train 200
```

mumax3 has no arrays, so the input sequence is unrolled into the script
(one block per reservoir step; `step` and `B_loop` table columns tag every
row). Islands are built from `Cuboid`/`Cylinder` shapes with `.Repeat()` so
they wrap across the periodic box exactly as in the mumax⁺ mask.

## Cost

A 2×2-unit-cell supercell (8 islands, 320×320×17 cells) needs, per input step,
roughly 60 energy minimisations (bipolar 30 mT loop, 1 mT steps) and one 26 ns
dynamic run at α = 0.001 – of order one to a few minutes on a modern GPU, i.e.
several hours for a 400-point dataset. Reduce cost with `--loop-step 2e-3`,
`--n-cells 1 1`, a shorter `--fmr-duration` (coarser frequency resolution) or
`--cell-xy 10e-9` for exploration; the reservoir runner checkpoints and resumes.

## Notes and choices worth knowing

* **Why disorder:** in a perfectly periodic supercell all islands switch at the
  same field and the reservoir response becomes a step function of the input.
  The 2 % size disorder (both layers share the footprint) gives a spread of
  switching fields as the lithographic "quenched disorder" does in the experiment.
* **Region-averaged m(t)** is what is stored (cheap). Modes with zero net moment
  in a layer are invisible to it, as they are to a CPW. Save full `m` frames
  (`--save-states`, or `sim.get_magnetization()`) if you want spatial mode maps.
* **mumax3 `sinc`** is sin(x)/x; the scripts use `sinc(2*pi*f_cut*(t-t0))`.
  numpy's `sinc` is normalised, so the mumax⁺ driver uses `np.sinc(2*f_cut*(t-t0))`.
* **Vortex seeding:** `geometry.vortex_magnetization` seeds CW/ACW vortices in
  chosen layers so you can start from vortex-rich microstates, as in the paper's
  ±30 mT minor-loop protocol along the symmetry-broken axis.
* **The toy reservoir** in `mock_reservoir.py` is *not* micromagnetics. It is a
  hysteretic macrospin-plus-dipole cartoon with Kittel-like Lorentzian "spectra"
  whose only purpose is exercising the pipeline and readout code in seconds.

## References

* T. Dion, K. D. Stenning, A. Vanstone *et al.*, "Ultrastrong magnon-magnon coupling and chiral
  spin-texture control in a dipolar 3D multilayered artificial spin-vortex ice",
  Nat. Commun. 15, 4077 (2024).
* J. C. Gartside, K. D. Stenning, A. Vanstone *et al.*, "Reconfigurable training and reservoir
  computing in an artificial spin-vortex ice via spin-wave fingerprinting", Nat. Nanotechnol. 17, 460 (2022).
* A. Vansteenkiste *et al.*, "The design and verification of MuMax3", AIP Adv. 4, 107133 (2014).
* mumax⁺: <https://github.com/mumax/plus>, npj Comput. Mater. (2025), doi:10.1038/s41524-025-01893-y.

## Workflow C – Google Colab from the terminal (Colab CLI)

No local GPU? The [Colab CLI](https://github.com/googlecolab/google-colab-cli)
rents a Colab GPU VM from your shell (`uv tool install google-colab-cli`, then
`colab sessions` once to log in). `colab/run_remote.sh` wraps the whole flow:

```bash
colab/run_remote.sh setup T4                       # new VM, upload repo, pip install mumaxplus, smoke test
colab/run_remote.sh sweep --b-start=-30e-3 --b-stop 60e-3 --n-cells 1 1 --out=runs/sweep
colab/run_remote.sh reservoir --task mackey_glass_10 --n 200 --n-cells 2 2 --out=runs/mg10
python scripts/train_readout.py runs/colab/mg10_spectra.npz --n-train 100 --plot mg10.png
colab/run_remote.sh stop                           # the VM is billed until you stop it
```

The kernel state persists between calls, results are downloaded to `runs/colab/`,
and `colab/run_remote.sh sync` re-uploads the repo after local edits.

Practical notes from running this (Colab CLI 0.6.0):

* The CLI's dependency `jupyter-kernel-client` 1.0 renamed a class the CLI
  imports; pin it: `uv pip install --python <cli venv> "jupyter-kernel-client<1"`.
* `colab exec` holds a WebSocket and the kernel runs cells serially, so long
  jobs are launched detached (`start`) and polled (`log`/`wait`); `exec` gets
  `--timeout 86400` (the wrapper does this).
* The runtime-proxy token cached per session lives 1 h. If the keep-alive
  daemon dies (e.g. your terminal/container sleeps) the next command 404s and
  the CLI *prunes* the session although the VM is still assigned and billing
  (`colab sessions` shows it as `[?]`). `colab/run_remote.sh adopt spinice=gpu-t4`
  re-attaches to the running kernel and restarts the daemon. Detached jobs
  survive all of this because they are ordinary processes on the VM.
