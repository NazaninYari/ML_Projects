import csv
import pprint
import numpy as np

EPS = 1e-6


def read_cancer():
    """Reading breast cancer data and converting the values to float and class names from 2 and 4 to 0, 1. We also
    remove the first column which is id numbers """
    with open("/Users/nazanin/Documents/Machine Learning/Project4_NazaninYari/input/breast-cancer-wisconsin.csv",
              'r') as dataset1:
        csv_reader = csv.reader(dataset1)
        data_set = []
        for row in csv_reader:
            tmp_row = []
            cls = row[-1]
            # Converting all numbers from string to float
            for value in row[1:-1]:
                if value == '?':
                    # replaced with mode or most commonly occuring value
                    tmp_row.append(1)
                else:
                    tmp_row.append(float(value))

            # Converting the class names to 0 and 1
            if cls == '2':
                tmp_row.append(0)
            elif cls == '4':
                tmp_row.append(1)
            data_set.append(tmp_row)
    return np.array(data_set)


def read_glass():
    """Reading glass data and converting the values to float and class names from 1 through 7 to 0 through 6. We also
        remove the first column which is id numbers """
    with open("/Users/nazanin/Documents/Machine Learning/Project4_NazaninYari/input/glass.csv",
              'r') as dataset2:
        csv_reader = csv.reader(dataset2)
        data_set = []
        for row in csv_reader:
            tmp_row = []
            cls = row[-1]
            # Converting all numbers from string to float
            for value in row[1:-1]:
                tmp_row.append(float(value))

            # Converting the class names to [1,2,3,5,6,7] -> [0,1,2,3,4,5]. There is no class 4 in this dataset
            if cls in ('1', '2', '3'):
                tmp_row.append(float(cls) - 1)
            else:
                tmp_row.append(float(cls) - 2)
            data_set.append(tmp_row)
    return np.array(data_set)


def data_stats(dataset):
    """Calculating mean, covariance and variance for each class"""

    # number of features
    num_cols = dataset.shape[1] - 1

    # The unique numbers in the last column of the dataset (which is class)
    class_num = np.unique(dataset[:, -1], return_counts=False)

    # Initializing means, std and cov matrices
    means = np.zeros((len(class_num), num_cols))
    std = np.zeros((len(class_num), num_cols))
    cov = np.zeros((len(class_num), num_cols, num_cols))

    # numpy array (matrix) have been used to store data stats with row number equal to the class number
    for cls in class_num:
        data_subset = dataset[np.where(dataset[:, -1] == cls)]
        means[int(cls), :] = np.mean(data_subset[:, :-1], axis=0)
        std[int(cls), :] = np.std(data_subset[:, :-1], axis=0)
        cov[int(cls), :] = np.cov(data_subset[:, :-1].T)

    return means, std, cov


def standardize_data(dataset):
    """standardizing dataset to center at zero with variance 1"""

    means, std, _ = data_stats(dataset)

    # The unique numbers in the last column of the dataset (which is class)
    class_num = np.unique(dataset[:, -1], return_counts=False)

    # Initializing a zero matrix to store the standardized data
    standardized_data = np.zeros(dataset.shape)

    for i in class_num:
        class_i_rows = np.where(dataset[:, -1] == i)

        # A subset of the dataset where the class=i
        data_subset = dataset[class_i_rows]

        # Standardizing the dataset by subtracting mean and dividing by variance
        standardized_data[class_i_rows, :-1] = (data_subset[:, :-1] - means[int(i), :]) / (std[int(i), :] + EPS)
        standardized_data[class_i_rows, -1] = i

    return standardized_data


def main():
    glass_data = read_glass()
    cancer_data = read_cancer()
    means, std, cov = data_stats(cancer_data)
    print(cov)


if __name__ == '__main__':
    main()
