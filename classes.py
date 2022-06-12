import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpp
from dataclasses import dataclass, field
import json


__author__ = [
    {
        'name': 'Jacob Jeffries',
        'orcid': '0000-0002-8852-6765'
    }
]
__version__ = 'FILL IN LATER'


@dataclass
class Atom:

    position: np.ndarray
    atom_type: str

    def square_distance_to(self, other) -> float:

        return np.sum((self.position - other.position) ** 2)

    def get_nearest_neighbors(self, popped_list: list, num_neighbors: int) -> list:
        
        sq_distances = [self.square_distance_to(atom) for atom in popped_list]
        tuples = sorted(zip(sq_distances, popped_list))[:num_neighbors]
        neighboring_atoms = [t[1] for t in tuples]
        
        return neighboring_atoms

    def __lt__(self, other) -> bool:

        return self.atom_type < other.atom_type

    def __hash__(self) -> int:

        return hash(self.atom_type)


def effective_volume_integrand(target_atom: Atom,
                               neighboring_atom_list: list[Atom],
                               min_potential_energy: float,
                               beta: float,
                               pairwise_interaction: callable,
                               interaction_params: dict) -> float:

    potential_energy = 0.0
    for other_atom in neighboring_atom_list:

        potential_energy += pairwise_interaction(target_atom,
                                                 other_atom,
                                                 interaction_params)

    return np.exp(-beta * potential_energy + beta * min_potential_energy)


@dataclass
class LocalEnvironment:

    center_atom: Atom
    neighboring_atoms: list[Atom]
    effective_volume: float = None
    effective_volume_std: float = None

    def __eq__(self, other) -> bool:

        if self.center_atom.atom_type != other.center_atom.atom_type:
            return False

        for first_atom, second_atom in zip(sorted(self.neighboring_atoms), sorted(other.neighboring_atoms)):
            if first_atom.atom_type != second_atom.atom_type:
                return False

        return True

    def __hash__(self) -> int:

        neighbors_tuple = sorted(self.neighboring_atoms)
        tuple_to_hash = (*neighbors_tuple, self.center_atom)

        return hash(tuple_to_hash)

    def compute_effective_volume(self,
                                 radius: float,
                                 min_potential_energy: float,
                                 beta: float,
                                 pairwise_interaction: callable,
                                 interaction_params: dict,
                                 sample_size: int = 1_000) -> float:

        function_values = [0 for _ in range(sample_size)]

        for index in range(sample_size):
            position = np.random.uniform(low=-radius, high=radius, size=3)
            temp_atom = Atom(position=position, atom_type=self.center_atom.atom_type)
            function_value = effective_volume_integrand(target_atom=temp_atom,
                                                        neighboring_atom_list=self.neighboring_atoms,
                                                        min_potential_energy=min_potential_energy,
                                                        beta=beta,
                                                        pairwise_interaction=pairwise_interaction,
                                                        interaction_params=interaction_params)
            function_values[index] = function_value

        mean = np.mean(function_values)
        std = np.std(function_values)
        volume = 4 / 3 * np.pi * radius ** 3

        self.effective_volume = volume * mean
        self.effective_volume_std = volume * std

    def __repr__(self) -> str:

        if self.effective_volume and self.effective_volume_std:
            return f'Center Atom: {self.center_atom.atom_type}\n' \
                   f'Neighboring Atoms: {[atom.atom_type for atom in self.neighboring_atoms]}\n' \
                   f'Effective Volume: {self.effective_volume} ± {self.effective_volume_std}\n'

        return f'Center Atom: {self.center_atom.atom_type}\n' \
               f'Neighboring Atoms: {[atom.atom_type for atom in self.neighboring_atoms]}\n'


@dataclass
class Atoms:

    atom_list: list[Atom]
    environments: list[LocalEnvironment] = None

    @staticmethod
    def pop_item(lst: list, index: int) -> list:

        return [item for j, item in enumerate(lst) if j != index]

    @staticmethod
    def get_units_dicts(file_name: str ='units.json') -> dict:

        with open(file_name, 'r') as file:

            dictionary = json.load(file)

        return dictionary

    def calculate_environments(self,
                               num_neighbors: int = 16,
                               log_file: str = 'environment.log') -> None:

        environments = [None for _ in self.atom_list]

        with open(log_file, 'w') as file:

            for index, center_atom in enumerate(self.atom_list):
                popped_list = self.pop_item(self.atom_list, index)
                neighboring_atoms = center_atom.get_nearest_neighbors(popped_list=popped_list,
                                                                      num_neighbors=num_neighbors)
                environments[index] = LocalEnvironment(center_atom=center_atom,
                                                       neighboring_atoms=neighboring_atoms)
                output = f'Environment of atom {index + 1} calculated and stored'
                print(output)
                file.write(f'{output}\n')

        self.environments = environments
        
    def get_blocks(self):
        
        if not self.environments:
            self.calculate_environments()
            
        unique_environments = set(self.environments)
        block_sizes = dict((unique_env, 0) for unique_env in unique_environments)
        for env in self.environments:
            for unique_env in unique_environments:
                if unique_env != env:
                    continue
                block_sizes[unique_env] += 1
                
        if len(self.atom_list) != sum(value for key, value in block_sizes.items()):
            raise ValueError('check block calculation')
        
        return block_sizes

    def plot(self,
             num_dimensions: int,
             units: str,
             palette_name: str ='Set2',
             save: bool =False,
             plot_type: str ='configuration',
             file_name: str = 'plot.png',
             **scatter_kwargs) -> None:

        if num_dimensions != 2 and num_dimensions != 3:
            raise ValueError('only 2 or 3 dimensions allowed')

        positions = np.array([atom.position for atom in self.atom_list]).T

        if positions.shape[0] != num_dimensions:
            raise ValueError('mismatched number of dimensions')

        unit_dict = self.get_units_dicts().get(units)
        length_unit = unit_dict.get('distance')

        plt.close()

        fig = plt.figure()

        if num_dimensions == 3:
            ax = fig.add_subplot(projection='3d')
            ax.set_zlabel(f'z ({length_unit})')
        else:
            ax = fig.add_subplot()
            ax.set_aspect(1)

        ax.set_xlabel(f'x ({length_unit})')
        ax.set_ylabel(f'y ({length_unit})')

        if plot_type == 'configuration':

            type_list = [atom.atom_type for atom in self.atom_list]
            unique_types = set(type_list)
            num_types = len(unique_types)
            palette = sns.color_palette(palette_name, n_colors=num_types)

            type_dictionary = dict(zip(unique_types, palette))
            color_list = [type_dictionary.get(t) for t in type_list]
            title = 'Atom Types'
            handles = [mpp.Patch(color=p, label=t) for p, t in zip(palette, unique_types)]

        elif plot_type == 'environments':

            environment_tags = [hash(env) for env in self.environments]
            unique_types = set(environment_tags)
            labels = [k + 1 for k, _ in enumerate(unique_types)]
            palette = sns.color_palette(palette_name, n_colors=len(unique_types))

            type_dictionary = dict(zip(unique_types, palette))
            color_list = [type_dictionary.get(t) for t in environment_tags]
            title = 'Environment Types'
            handles = [mpp.Patch(color=p, label=l) for p, l in zip(palette, labels)]

        else:
            raise ValueError('invalid plot_type, pick configuration or environments')

        ax.scatter(*positions,
                   color=color_list,
                   linewidths=1.0,
                   edgecolors='black',
                   zorder=3,
                   **scatter_kwargs)
        plt.legend(title=title,
                   handles=handles,
                   loc='center left',
                   bbox_to_anchor=(1.1, 0.4))

        if save:
            plt.savefig(file_name, dpi=800, bbox_inches='tight')