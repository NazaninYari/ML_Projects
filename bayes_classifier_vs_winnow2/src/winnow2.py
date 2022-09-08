import pickle

import numpy as np

from bayes_classifier import data_train_test


def h_x(row, weight_list):
    """Calculating h(x)"""
    h = 0

    # If weight list was empty, set initial weights to 1
    if not weight_list:
        for i in range(len(row)):
            weight_list.append(1)

    for i in range(len(row)):
        h += row[i] * weight_list[i]
    return h


def updated_weight(row, alpha, initial_weight, predicted_value, desired_output):
    """Calculating updated weights: demote or promote after comparing h(x) to f(x)"""
    if predicted_value > desired_output:
        weight_list = demote(row, alpha, initial_weight)
    elif predicted_value < desired_output:
        weight_list = promote(row, alpha, initial_weight)
    else:
        weight_list = initial_weight

    return weight_list


def prediction(h, theta):
    """The prediction function (f(x))"""
    if h > theta:
        return 1
    else:
        return 0


def demote(row, alpha, weight_list):
    """demote function"""
    for i in range(len(row)):
        if row[i] == 1:
            weight_list[i] = weight_list[i] / alpha
        elif row[i] == 0:
            continue

    return weight_list


def promote(row, alpha, weight_list):
    """promote function"""
    for i in range(len(row)):
        if row[i] == 1:
            weight_list[i] = weight_list[i] * alpha
        elif row[i] == 0:
            continue

    return weight_list


def winnow_accuracy_test(x_test, y_test, x_train, y_train, alpha, theta):
    """Training model in order to calculate the final weight, and using final weight to test the test data set.
    The column with predicted value has been added to the last column of the comparison table"""
    # Training model using train data set
    train_row_num = x_train.shape[0]
    weight_list = []

    for row in range(train_row_num):
        h = h_x(x_train[row], weight_list)
        predicted_value = prediction(h, theta)
        weight_list = updated_weight(x_train[row], alpha, weight_list, predicted_value, y_train[row, 0])

    combined_x_y = np.concatenate((x_test, y_test), axis=1)
    tmp_vector = []
    test_row_num = x_test.shape[0]

    for row in range(test_row_num):
        h = h_x(x_test[row], weight_list)
        predicted_value = prediction(h, theta)
        tmp_vector.append(predicted_value)

    tmp_vector = np.reshape(tmp_vector, (test_row_num, 1))
    comparison_table = np.concatenate((combined_x_y, tmp_vector), axis=1)

    # Calculating accuracy of the prediction by comparing to the class column of y_train
    accuracy = 0
    for row in comparison_table:
        if row[-1] == row[-2]:
            accuracy += 1
    error_percentage = 1 - (accuracy / comparison_table.shape[0])

    return comparison_table, error_percentage


def main():
    # loading previously prepared data sets: boolean versions
    with open('data_sets.pkl', 'rb') as data_sets:
        blackboard_datasets = pickle.load(data_sets)
    part1_data_bool = blackboard_datasets['part1_data_bool']
    part2_data_bool = blackboard_datasets['part2_data_bool']
    part3_data_bool = blackboard_datasets['part3_data_bool']

    # loading previously prepared vote data set
    with open('vote_data_set.pkl', 'rb') as vote_data_sets:
        vote_data_set = pickle.load(vote_data_sets)
    vote_data = vote_data_set['vote_data']

    x_train_vote, x_test_vote, y_train_vote, y_test_vote = data_train_test(vote_data)
    x_train_1_bool, x_test_1_bool, y_train_1_bool, y_test_1_bool = data_train_test(part1_data_bool)
    x_train_2_bool, x_test_2_bool, y_train_2_bool, y_test_2_bool = data_train_test(part2_data_bool)
    x_train_3_bool, x_test_3_bool, y_train_3_bool, y_test_3_bool = data_train_test(part3_data_bool)

    # Writing comparison tables into output files. The accuracy percentage has been added to the last row of the table
    comparison_table_1_bool, error_1_bool = winnow_accuracy_test(x_test_1_bool, y_test_1_bool, x_train_1_bool,
                                                                 y_train_1_bool, alpha=2, theta=0.5)
    np.savetxt(
        'output_files/Winnow2/comparison_table_part1_bool.csv',
        comparison_table_1_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_1_bool}')

    comparison_table_2_bool, error_2_bool = winnow_accuracy_test(x_test_2_bool, y_test_2_bool, x_train_2_bool,
                                                                 y_train_2_bool, alpha=2, theta=0.5)
    np.savetxt(
        'output_files/Winnow2/comparison_table_part2_bool.csv',
        comparison_table_2_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_2_bool}')

    comparison_table_3_bool, error_3_bool = winnow_accuracy_test(x_test_3_bool, y_test_3_bool, x_train_3_bool,
                                                                 y_train_3_bool, alpha=2, theta=0.5)
    np.savetxt(
        'output_files/Winnow2/comparison_table_part3_bool.csv',
        comparison_table_3_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_3_bool}')

    comparison_table_vote, error_vote = winnow_accuracy_test(x_test_vote, y_test_vote, x_train_vote, y_train_vote,
                                                             alpha=2, theta=0.5)
    np.savetxt(
        'output_files/Winnow2/comparison_table_vote.csv',
        comparison_table_vote,
        delimiter=',', header='feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,'
                              'feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,'
                              'class,predicted_class', fmt='%0d',
        footer=f'Error:{error_vote}')


if __name__ == '__main__':
    main()
