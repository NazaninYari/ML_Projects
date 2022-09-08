import csv
import pickle
import numpy as np

# letter class names -> numbers
cls_dict = {'G': 0, 'H': 1}
EPS = 1e-6


def read_data(path):
    """Reading data and converting the values to float and class names to 0, 1"""
    with open(path, 'r') as dataset:
        csv_reader = csv.reader(dataset)
        data_set = []
        for row in csv_reader:
            tmp_row = []
            cls = row[-1]
            # Converting all numbers from string to float
            for value in row[:-1]:
                tmp_row.append(float(value))

            # Converting the class names to 0 and 1
            if cls in cls_dict.keys():
                tmp_row.append(cls_dict[cls])
            else:
                tmp_row.append(float(cls))
            data_set.append(tmp_row)
    return np.array(data_set)


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


def standardize_data(dataset):
    """standardizing dataset to center at zero with variance 1"""
    _, means, std = data_stats(dataset)
    # The unique numbers in the last column of the dataset (which is class)
    class_num = np.unique(dataset[:, -1], return_counts=False)
    # Initializing a zero matrix to store the standardized data
    standardized_data = np.zeros(dataset.shape)

    for i in class_num:
        class_i_rows = np.where(dataset[:, -1] == i)
        # A subset of the dataset where the class=i
        data_subset = dataset[class_i_rows]
        # Standardizing the dataset by subtracting mean and dividing by variance
        standardized_data[class_i_rows, :-1] = (data_subset[:, :-1] - means[i]) / (std[i] + EPS)
        standardized_data[class_i_rows, -1] = i
    return standardized_data


def write_data(output_path, array_to_write):
    """writing into files"""
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(array_to_write)


def main():
    synthetic_data = read_data(
        '/Users/nazanin/Documents/Machine Learning/Project2_NazaninYari/input_dataset/proj2.csv')
    spambase_data = read_data(
        '/Users/nazanin/Documents/Machine Learning/Project2_NazaninYari/input_dataset/spambase.csv')
    synthetic_4000_data = read_data(
        '/Users/nazanin/Documents/Machine Learning/Project2_NazaninYari/input_dataset/synthetic_4000.csv')

    # Mean and covariance of datasets
    synthetic_cov, synthetic_means, synthetic_std = data_stats(synthetic_data)
    spam_cov, spam_means, spam_std = data_stats(spambase_data)

    # Standardized Dataset
    standardized_synthetic = standardize_data(synthetic_data)
    standardized_spambase = standardize_data(spambase_data)
    standardized_synthetic_4000 = standardize_data(synthetic_4000_data)

    # Writing standardized dataset into files
    write_data('/Users/nazanin/Documents/Machine Learning/Project2_NazaninYari/output_files/standardized_synthetic.csv',
               standardized_synthetic)
    write_data('/Users/nazanin/Documents/Machine Learning/Project2_NazaninYari/output_files/standardized_spambase.csv',
               standardized_spambase)
    write_data('/Users/nazanin/Documents/Machine '
               'Learning/Project2_NazaninYari/output_files/standardized_synthetic_4000.csv',
               standardized_synthetic_4000)

    # dumping data sets into output to be used in the other files
    proj2_dataset = {
        'synthetic_data': synthetic_data,
        'spambase_data': spambase_data,
        'synthetic_4000': synthetic_4000_data,
        'standardized_synthetic': standardized_synthetic,
        'standardized_spambase': standardized_spambase,
        'standardized_synthetic_4000': standardized_synthetic_4000

    }
    with open('proj2_dataset.pkl', 'wb') as output:
        pickle.dump(proj2_dataset, output)


if __name__ == '__main__':
    main()
