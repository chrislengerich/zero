# Split a dataset file into train and validation sets.
from __future__ import print_function

def split(file_path, file_path_train, file_path_val, validation_fraction=0.3):
    with open(file_path) as f:
        lines = f.readlines()

    split = int(len(lines) * validation_fraction)
    with open(file_path_val, 'w') as f:
        for l in lines[:split]:
            print(l.strip(), file=f)

    with open(file_path_train, 'w') as f:
        for l in lines[split:]:
            print(l.strip(), file=f)


if __name__ == "__main__":
    split("data/trainval_car", "data/train_car", "data/val_car")