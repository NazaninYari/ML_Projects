import math
import pickle
import warnings

from Project2.bayes_classifier import data_train_test, bayes_error

warnings.filterwarnings('error')


def sequential_forward(dataset):
    """Finding the best features and calculating the error"""
    # number of rows and features in data set
    row_num, col_num = dataset.shape

    # a list of all features in data set(excluding class column)
    remaining_features = list(range(0, col_num - 1))

    # initializing best features list
    selected_features = []

    # list of errors
    error_list = [math.inf]

    while remaining_features:

        best_error = math.inf

        for i, feature_i in enumerate(remaining_features):
            # The column index of previously selected features,as well as the new feature
            column_slice = selected_features + [feature_i, -1]

            data_subset = dataset[:, column_slice]
            x_train, x_test, y_train, y_test = data_train_test(data_subset)
            error = bayes_error(x_test, y_test, x_train, y_train)

            if error < best_error:
                best_error = error
                best_feature_idx = feature_i
                best_i = i
        if best_error < error_list[-1]:
            error_list.append(best_error)
            selected_features.append(best_feature_idx)
            remaining_features.pop(best_i)
        else:
            break

    return error_list, selected_features


def main():
    # loading previously prepared data sets
    with open('proj2_dataset.pkl', 'rb') as data_sets:
        datasets = pickle.load(data_sets)
    synthetic_data = datasets['synthetic_data']
    spambase_data = datasets['spambase_data']
    synthetic_4000 = datasets['synthetic_4000']
    standardized_synthetic = datasets['standardized_synthetic']
    standardized_spambase = datasets['standardized_spambase']
    standardized_synthetic_4000 = datasets['standardized_synthetic_4000']

    print(sequential_forward(standardized_synthetic_4000))


if __name__ == '__main__':
    main()
