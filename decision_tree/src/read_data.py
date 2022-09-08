import numpy as np
import csv


def read_data(path):
    """Reading data and storing them into a numpy array"""
    dataset = []
    with open(path, 'r') as dataset_file:
        csv_reader = csv.reader(dataset_file)
        for row in csv_reader:
            row_tmp = []
            for value in row:
                row_tmp.append(value)
            dataset.append(row_tmp)
    return dataset


def split_to_x_y(dataset):
    """Splits into input and output and converts outputs to indices."""
    dataset = np.array(dataset)
    x = dataset[:, :-1]
    y = dataset[:, -1]
    y_id_to_label = dict(enumerate(np.unique(y)))
    y_label_to_id = {label: idx for idx, label in y_id_to_label.items()}
    y_ids = np.array([y_label_to_id[label] for label in y])
    y_ids = np.expand_dims(y_ids, axis=1)  # to make y 2d.
    return x, y_ids, y_id_to_label

