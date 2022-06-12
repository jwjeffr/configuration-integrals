import json

def get_dictionary():
    # metal units

    dictionary = {
        'LJ': {
            'Ar-Ar': {
                'sigma': 3.4,
                'dispersion': 0.0103434293
            },
            'Ar-ArT': {  # tagging an argon atom for testing
                'sigma': 3.4,
                'dispersion': 0.0103434293
            },
            'He-He': {
                'sigma': 2.56,
                'dispersion': 0.00088092
            },
            'Ne-Ne': {
                'sigma': 2.75,
                'dispersion': 0.00306855069
            },
            'Kr-Kr': {
                'sigma': 3.83,
                'dispersion': 0.01413602
            },
            'Xe-Xe': {
                'sigma': 4.06,
                'dispersion': 0.0197387109
            }
        }
    }
    
    return dictionary


def main():

    with open('interaction_dictionaries.json', 'w') as file:
        json.dump(dictionary, file)


if __name__ == '__main__':

    main()
