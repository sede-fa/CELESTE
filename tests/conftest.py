"""Shared fixtures of the CELESTE test suite."""
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def golden():
    """Reference property values recorded with the original (Numba) implementation."""
    with (DATA / "golden_thermo.json").open() as f:
        return json.load(f)


@pytest.fixture
def syngas_x():
    """A wet syngas in the species order of media.Fuel_CH4: H2, CH4, CO2, CO, H2O, N2."""
    return np.array([0.6, 0.05, 0.1, 0.05, 0.2, 1e-8])


def element_flow(flange, element):
    """Molar flow of an element through a flange, mol/s."""
    from celeste.jx import nasa
    per_species = np.array([nasa.elements(str(s).replace('(L)', '') if str(s) not in nasa.species_names() else str(s)).get(element, 0.0)
                            for s in flange.Medium.species])
    return flange.n*float(np.dot(per_species, flange.Xi))
