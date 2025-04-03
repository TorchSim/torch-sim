import torch
from pymatgen.core.composition import Composition

from torch_sim.workflows.a2c import (
    get_diameter,
    get_diameter_matrix,
    get_subcells_to_crystallize,
    get_target_temperature,
    subcells_to_structures,
)


def test_get_diameter():
    # Test with a single element (Cu)
    comp = Composition("Cu")
    diameter = get_diameter(comp)
    assert diameter > 0

    # Test with a binary compound (Fe2O3)
    comp = Composition("Fe2O3")
    diameter = get_diameter(comp)
    assert diameter > 0

    # Test with a more complex composition
    comp = Composition("Li3PS4")
    diameter = get_diameter(comp)
    assert diameter > 0


def test_get_diameter_matrix(device: torch.device):
    # Test with a single element (Cu)
    comp = Composition("Cu")
    matrix = get_diameter_matrix(comp, device=device)

    # Should be a 1x1 matrix for single element
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] > 0

    # Test with a binary compound (Fe2O3)
    comp = Composition("Fe2O3")
    matrix = get_diameter_matrix(comp, device=device)

    # Should be a 2x2 matrix for Fe and O
    assert matrix.shape == (2, 2)
    assert torch.all(matrix > 0)

    # Matrix should be symmetric
    assert torch.allclose(matrix, matrix.T)


def test_get_subcells_to_crystallize(device: torch.device):
    # Create test data
    frac_positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
        device=device,
    )
    species = ["Cu", "Cu", "O", "O", "O"]

    # Test without composition restrictions
    candidates = get_subcells_to_crystallize(
        frac_positions, species, d_frac=0.5, n_min=1, n_max=5
    )

    # Should find at least some subcells
    assert len(candidates) > 0

    # Check structure of returned candidates
    for ids, lower, upper in candidates:
        assert isinstance(ids, torch.Tensor)
        assert isinstance(lower, torch.Tensor)
        assert isinstance(upper, torch.Tensor)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    # Test with composition restrictions
    candidates = get_subcells_to_crystallize(
        frac_positions,
        species,
        d_frac=0.5,
        n_min=1,
        n_max=5,
        restrict_to_compositions=["CuO"],
    )

    # Should find subcells with Cu:O ratio of 1:1
    for ids, _, _ in candidates:
        subcell_species = [species[int(i)] for i in ids.cpu().numpy()]
        cu_count = subcell_species.count("Cu")
        o_count = subcell_species.count("O")
        assert cu_count == o_count

    # Test with max_coeff and elements
    candidates = get_subcells_to_crystallize(
        frac_positions,
        species,
        d_frac=0.5,
        n_min=1,
        n_max=5,
        max_coeff=2,
        elements=["Cu", "O"],
    )

    # Should not include compositions with coefficients > 2
    assert len(candidates) > 0


def test_subcells_to_structures(device: torch.device):
    # Create test data
    frac_positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
        device=device,
    )
    cell = torch.tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], device=device
    )
    species = ["Cu", "Cu", "O", "O", "O"]

    # Get subcell candidates
    candidates = get_subcells_to_crystallize(
        frac_positions, species, d_frac=0.5, n_min=1, n_max=5
    )

    # Convert to structures
    structures = subcells_to_structures(candidates, frac_positions, cell, species)

    # Check output format
    assert len(structures) == len(candidates)
    for pos, subcell, spec in structures:
        assert isinstance(pos, torch.Tensor)
        assert isinstance(subcell, torch.Tensor)
        assert isinstance(spec, list)
        assert pos.shape[1] == 3  # 3D positions
        assert subcell.shape == (3, 3)  # 3x3 cell matrix
        assert all(isinstance(s, str) for s in spec)  # Species strings


def test_get_target_temperature():
    # Test equilibration phase (high temperature)
    temp = get_target_temperature(
        step=10, equi_steps=20, cool_steps=30, T_high=1000, T_low=300
    )
    assert temp == 1000

    # Test cooling phase (decreasing temperature)
    temp = get_target_temperature(
        step=30, equi_steps=20, cool_steps=30, T_high=1000, T_low=300
    )
    assert 300 < temp < 1000

    # Test at end of cooling
    temp = get_target_temperature(
        step=49, equi_steps=20, cool_steps=30, T_high=1000, T_low=300
    )
    assert temp == 300 + (1000 - 300) / 30  # Should be one step above T_low

    # Test final low temperature
    temp = get_target_temperature(
        step=60, equi_steps=20, cool_steps=30, T_high=1000, T_low=300
    )
    assert temp == 300
