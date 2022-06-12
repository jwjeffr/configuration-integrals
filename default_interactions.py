import classes
import numpy as np


def lj_interaction_homogeneous(first_atom, second_atom, interaction_params):

    sigma = interaction_params['sigma']
    dispersion = interaction_params['dispersion']
    distance = np.linalg.norm(first_atom.position - second_atom.position)

    return 4 * dispersion * ((sigma / distance) ** 12 - (sigma / distance) ** 6)


def lj_interaction_mixed(first_atom, second_atom, interaction_params):

    sigma = interaction_params[f'sigma-{first_atom.atom_type}-{second_atom.atom_type}']
    dispersion = interaction_params[f'dispersion-{first_atom.atom_type}-{second_atom.atom_type}']
    distance = np.linalg.norm(first_atom.position - second_atom.position)

    return 4 * dispersion * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
