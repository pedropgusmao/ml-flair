import argparse
import pickle
from pathlib import Path
from typing import Dict


def label_relationship_to_dict(label_rel_file: Path) -> Dict[str, str]:
    # Keeping the same labels_and_metadata.json labelling format
    rel_dict: Dict[str, Dict[str, int]] = {"labels": {}, "fine_grained_labels": {}}
    list_coarse = []
    list_fine = []
    with open(label_rel_file) as f:
        for line in f:
            (fine_grained, coarse_grained) = line.split(" -> ")
            list_coarse.append(coarse_grained.strip())
            list_fine.append(fine_grained.strip())

    list_coarse = sorted(set(list_coarse))
    list_fine = sorted(set(list_fine))

    for item in list_coarse:
        rel_dict["labels"][item] = list_coarse.index(item)
    for item in list_fine:
        rel_dict["fine_grained_labels"][item] = list_fine.index(item)

    return rel_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert label relationships to dictionary"
    )
    parser.add_argument(
        "--path_to_label_relationship",
        type=Path,
        help="Path to label relationships file",
        required=True,
    )
    args = parser.parse_args()
    save_path = args.path_to_label_relationship.parent / "flair_labels_to_index.pkl"
    rel_dict = label_relationship_to_dict(args.path_to_label_relationship)
    with open(save_path, "wb") as f:
        pickle.dump(rel_dict, f)
