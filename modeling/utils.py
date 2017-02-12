import json


def save_dict_to_file(dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(dict, f)
