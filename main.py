import classes
import default_interactions
import interaction_dictionaries
import numpy as np


def main():

    data = np.loadtxt('all_atoms.0.dump', skiprows=9, dtype=float)
    positions = data[:, 2:5][:50]

    atom_list = []
    for position in positions:
        atom_type = np.random.choice(('Ar', 'ArT'), p=(0.9, 0.1))
        atom = classes.Atom(position=position,
                            atom_type=atom_type)
        atom_list.append(atom)

    atoms = classes.Atoms(atom_list=atom_list)
    
    block_sizes = atoms.get_blocks()

    dictionary = interaction_dictionaries.get_dictionary()
    interaction_params = dictionary['LJ']['Ar-Ar']
    pairwise_interaction = default_interactions.lj_interaction_homogeneous
    for env, num in block_sizes.items():
        env.compute_effective_volume(radius=10.0,
                                     min_potential_energy=-100.0,
                                     beta=150.0,
                                     pairwise_interaction=pairwise_interaction,
                                     interaction_params=interaction_params,
                                     sample_size=100)
        print(f'Effective volume = {env.effective_volume}, block size = {num}')


if __name__ == '__main__':

    main()
