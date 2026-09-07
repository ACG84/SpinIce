"""Control: the ASVI lattice automaton reduced to flatspin-like single macrospins (rigid bilayer, two
states) with point-dipole stray fields and flatspin's numbers (B_c = 30 mT -> 15 mT threshold at 45 deg),
leak protocol with a zero-field relaxation before the readout.  Does the automaton's dynamics produce
lattice memory in the regime where flatspin does?

    python scripts/macro_control.py 7 9 12        # coupling multipliers (x9 ~ flatspin alpha 0.005)
    BC=0.072 python scripts/macro_control.py 1 2   # the ASVI's own 36 mT threshold
"""
import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from asvi_rc.lattice_automaton import ASVILattice
from scripts.flatspin_rc import ridge_r2

class Macro(ASVILattice):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        i_pp = self.labels.index(('+', '+')); i_mm = self.labels.index(('-', '-'))
        keep = [i_pp, i_mm]
        self.labels = [self.labels[i] for i in keep]; self.E = np.zeros(2)
        self.m_axis = self.m_axis[keep]; self.M_layer = self.M_layer[keep]; self.Q_layer = self.Q_layer[keep]
        self.Mtot = self.Mtot[keep]; self.Qs = self.Qs[keep]; self.K = 2
        self.M_layer[:, 0] = self.Mtot                         # whole moment switches as "layer 0"
        self.trans = [[(1, 0)], [(0, 0)]]; self.ground = 0; self.reset()

def run(bc, window, leak, jitter, coupling, det, n=800, seed=0, cells=(6, 6), update="sequential"):
    lat = Macro('docs/data/np_single_10/catalogue.json', [bc, bc], cells, None, 2e-3, seed=seed, disorder=0.05,
                coupling=coupling, switching="sw", update=update, field_model="dipole")
    rng = np.random.default_rng(100 + seed); u = rng.random(n)
    d = np.array([1, 1]) / np.sqrt(2); X = []; fl = []; keys = set()
    lat.relax(0.2 * d, sample=False); sf = lat.stray_field("centre"); ax = lat.axis
    par = abs(np.einsum('ik,ik->i', ax, sf)).mean(); perp = abs(sf[:, 0] * ax[:, 1] - sf[:, 1] * ax[:, 0]).mean()
    lat.reset()
    for t in range(n):
        L = leak + jitter * rng.uniform(-1, 1); lat.relax(-L * d, sample=not det)
        h = window[0] + (window[1] - window[0]) * u[t]; fl.append(lat.relax(h * d, sample=not det))
        lat.relax(0.0 * d, sample=not det)                                   # zero-field relaxation before the readout, as in flatspin_rc
        X.append(lat.features()); keys.add(lat.state_key()) if t > n // 2 else None
    r2, mc = ridge_r2(np.array(X), u, k_max=6, alpha=1e-2)
    return r2.round(2), round(mc, 2), np.mean(fl), len(keys), par * 1e3, perp * 1e3

if __name__ == "__main__":
    import sys, os
    bc = float(os.environ.get("BC", "0.030"))          # 30 mT -> 15 mT threshold at 45 deg (ideal SW), like flatspin hc = 30 mT
    print(f"### macrospin control with flatspin's numbers + zero-field relax before readout: point dipoles, Bc {bc*1e3:.0f} mT ({bc*0.5e3:.1f} mT at 45 deg), 6x6")
    for coupling in [float(x) for x in sys.argv[1:]] or [6.0]:
        for update in ["sequential", "parallel"]:
            for window, leak, jitter in [((45e-3, 75e-3), 30e-3, 4.5e-3), ((36e-3, 66e-3), 30e-3, 4.5e-3), ((60e-3, 105e-3), 45e-3, 6e-3), ((30e-3, 54e-3), 24e-3, 4e-3)]:
                for det in [True, False]:
                    t0 = time.time(); r2, mc, fl, nk, par, perp = run(bc, window, leak, jitter, coupling, det, update=update)
                    print(f"x{coupling:g} {update:10s} stray par {par:.1f} perp {perp:.1f} mT | window {window[0]*1e3:.0f}-{window[1]*1e3:.0f} leak {leak*1e3:.0f}+-{jitter*1e3:.1f} {'det' if det else 'sampled':7s} R2 {r2} MC {mc} flips {fl:.1f} distinct {nk} ({time.time()-t0:.0f} s)", flush=True)
    print("### done")
