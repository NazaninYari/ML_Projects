import csv
import pickle
import pprint

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# letter class names -> numbers
cls_dict = {'A': 0, 'B': 1, 'C': 0, 'D': 1, 'E': 0, 'F': 1}


def read_data(path):
    """Reading data and converting the values to float and class names to 0, 1"""
    with open(path, 'r') as dataset:
        csv_reader = csv.reader(dataset)
        data_set = []

        for row in csv_reader:
            tmp_row = []
            for value in row[:-1]:
                tmp_row.append(float(value))
            tmp_row.append(cls_dict[row[-1]])
            data_set.append(tmp_row)

    return np.array(data_set)


def bool_data(dataset):
    """Converting the data to boolean using mean -> 0 if less than mean, 1 if greater than mean"""
    row_num, col_num = dataset.shape
    data_bool = []

    # means of each feature of the data set
    means = np.mean(dataset, axis=0)

    for row in range(row_num):
        cls = int(dataset[row][-1])
        bool_row = []
        for col in range(col_num - 1):
            if dataset[row][col] < means[col]:
                bool_row.append(0)
            else:
                bool_row.append(1)
        bool_row.append(cls)
        data_bool.append(bool_row)

    return np.array(data_bool)


def data_stats(dataset):
    """Calculating mean, covariance and variance for each class"""

    # Dictionaries have been used to store the mean, covariance and standard deviation where the keys are classes
    means = {}
    cov = {}
    std = {}

    # The unique numbers in the last column of the dataset (which is class)
    class_num = np.unique(dataset[:, -1], return_counts=False)

    for i in class_num:
        # A subset of the dataset where the class=i
        data_subset = dataset[np.where(dataset[:, -1] == i)]
        # A subset of the dataset excluding the last column
        data_subset_features = data_subset[:, :-1]
        means[i] = np.mean(data_subset_features, axis=0)
        cov[i] = np.cov(data_subset_features.T)
        std[i] = np.std(data_subset_features, axis=0)
    return cov, means, std


def plot_data(dataset):
    """Plotting the datasets"""
    for c in np.unique(dataset[:, -1]).astype(int):
        c_rows = np.where(dataset[:, -1] == c)
        feature_1 = dataset[c_rows, 0]
        feature_2 = dataset[c_rows, 1]
        plt.scatter(x=feature_1, y=feature_2, s=4, label=f'class {c}')


def show_plot(dataset, x_axis, y_axis, title):
    """Showing the plots"""
    plot_data(dataset)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def write_data(output_path, array_to_write):
    """ Writing into file"""
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(array_to_write)


def main():
    # Reading 3 Blackboard files
    part1_data = read_data('input_files/proj1_part1.csv')
    part2_data = read_data('input_files/proj1_part2.csv')
    part3_data = read_data('input_files/proj1_part3.csv')

    part1_data_bool = bool_data(part1_data)
    part2_data_bool = bool_data(part2_data)
    part3_data_bool = bool_data(part3_data)

    # Calculating covarinace, mean and standard deviation for the datasets
    part1_cov, part1_means, part1_std = data_stats(part1_data)
    part2_cov, part2_means, part2_std = data_stats(part2_data)
    part3_cov, part3_means, part3_std = data_stats(part3_data)
    part1_bool_cov, part1_bool_means, part1_bool_std = data_stats(part1_data_bool)
    part2_bool_cov, part2_bool_means, part2_bool_std = data_stats(part2_data_bool)
    part3_bool_cov, part3_bool_means, part3_bool_std = data_stats(part3_data_bool)

    # Writing the 3 converted data arrays into files
    write_data('input_files/boolean_files/bool_data_part1.csv', part1_data_bool)
    write_data('input_files/boolean_files/bool_data_part2.csv', part2_data_bool)
    write_data('input_files/boolean_files/bool_data_part3.csv', part3_data_bool)

    # dumping datasets into output to be used in the Winnow2 and Bayes programs
    data_sets = {
        'part1_data': part1_data,
        'part2_data': part2_data,
        'part3_data': part3_data,
        'part1_data_bool': part1_data_bool,
        'part2_data_bool': part2_data_bool,
        'part3_data_bool': part3_data_bool
    }

    with open('data_sets.pkl', 'wb') as output:
        pickle.dump(data_sets, output)

    print("Covariance is:")
    pprint.pprint(part3_bool_cov)
    print('\nMean is: ')
    pprint.pprint(part3_bool_means)
    print('\nSandard Deviation is: ')
    pprint.pprint(part3_bool_std)

    show_plot(part1_data, 'X1', 'X2', 'Dataset: Part1')


if __name__ == '__main__':
    main()
