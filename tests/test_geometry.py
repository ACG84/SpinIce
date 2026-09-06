import math

import numpy as np
import pytest

from asvi_rc import ASVIParams, build_geometry
from asvi_rc.geometry import (build_islands, macrospin_magnetization, vortex_magnetization,
                              region_masks, region_id)


def small():
    return ASVIParams(n_cells=(1, 1), cell_xy=10e-9, cell_z=5e-9, disorder_sigma=0.0)


def test_grid_and_layers():
    p = small()
    assert p.lattice_constant == pytest.approx(800e-9)
    assert p.grid == (80, 80, 17)
    assert p.z_layers == ((0, 6), (13, 17))
    assert p.n_islands == 2


def test_incommensurate_cell_raises():
    with pytest.raises(ValueError):
        ASVIParams(cell_z=10e-9).grid      # 35 nm spacer is not a multiple of 10 nm


def test_islands_and_regions():
    p = small()
    islands = build_islands(p)
    assert len(islands) == 4
    assert sorted(i.region for i in islands) == [1, 2, 3, 4]
    assert region_id(0, 1) == 2
    top = [i for i in islands if i.layer == 1]
    bot = [i for i in islands if i.layer == 0]
    for t, b in zip(top, bot):
        assert t.cy - b.cy == pytest.approx(50e-9)


def test_geometry_volume_and_wrapping():
    p = small()
    mask, regions, islands = build_geometry(p)
    area = (p.length - p.width) * p.width + math.pi * (p.width / 2) ** 2
    expected = 2 * area * (p.t_bottom + p.t_top)
    vol = mask.sum() * p.cell_xy ** 2 * p.cell_z
    assert vol == pytest.approx(expected, rel=0.03)
    # spacer layers are empty
    assert not mask[6:13].any()
    # every region has the right number of layers of cells
    rm = region_masks(regions, islands)
    assert len(rm[1]) == 6 * (regions[0] == 1).sum()
    assert len(rm[2]) == 4 * (regions[13] == 2).sum()
    # an island shifted across the box edge is wrapped, not clipped
    p2 = small()
    islands2 = build_islands(p2)
    for isl in islands2:
        isl.cx += 300e-9
    mask2, _, _ = build_geometry(p2, islands2)
    assert mask2.sum() == mask.sum()


def test_magnetization_helpers():
    p = small()
    mask, regions, islands = build_geometry(p)
    m = macrospin_magnetization(p, regions, islands, {1: -1.0})
    assert m.shape == (3, 17, 80, 80)
    norms = np.linalg.norm(m, axis=0)
    assert np.allclose(norms[mask], 1) and np.allclose(norms[~mask], 0)
    assert m[0][regions == 1].mean() == pytest.approx(-1)
    assert m[1][regions == 3].mean() == pytest.approx(1)   # vertical island along +y
    mv = vortex_magnetization(p, regions, islands, chirality={2: 1})
    assert np.allclose(np.linalg.norm(mv, axis=0)[mask], 1)
    assert abs(mv[0][regions == 2].mean()) < 0.1 and abs(mv[1][regions == 2].mean()) < 0.1


def test_three_layer_stack_geometry():
    from asvi_rc import ASVIParams
    from asvi_rc.geometry import single_island, build_geometry, build_islands
    p = ASVIParams(cell_xy=10e-9, cell_z=5e-9, disorder_sigma=0.0, box_override=(640e-9, 320e-9),
                   pbc_repetitions=(0, 0, 0), extra_layers=((35e-9, 15e-9),))
    assert p.n_layers == 3 and p.thickness == 30e-9 + 35e-9 + 20e-9 + 35e-9 + 15e-9
    assert p.z_layers == ((0, 6), (13, 17), (24, 27))
    assert p.layer_offsets[2][1] > p.top_offset[1]                 # shadow offset grows with height
    isl = single_island(p)
    assert [i.region for i in isl] == [1, 2, 3]
    mask, regions, _ = build_geometry(p, isl)
    assert mask.shape[0] == 27 and set(np.unique(regions)) == {0, 1, 2, 3}
    assert regions[25].max() == 3 and regions[15].max() == 2 and regions[3].max() == 1
    # two-layer default unchanged, unit cell numbering cycles through the stack
    q = ASVIParams(cell_xy=10e-9, cell_z=5e-9, disorder_sigma=0.0, n_cells=(1, 1))
    assert [i.region for i in build_islands(q)] == [1, 2, 3, 4] and q.n_layers == 2
