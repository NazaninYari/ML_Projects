import pickle
import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

EPS = 1e-6

np.random.seed(0)
warnings.filterwarnings('error')


def data_train_test(data_set: np.array):
    """Creating test and train data sets"""
    # Shuffle the data
    np.random.shuffle(data_set)

    # Including all rows and columns except the last column (class)
    x = data_set[:, :-1]

    # vector of classes (last column)
    y = data_set[:, -1]

    y = np.expand_dims(y, axis=1)

    # Selecting 67% (2/3) of observation as training data and 33% (1/3) as test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    return x_train, x_test, y_train, y_test


def train_stats(x_train, y_train):
    """Calculating train set statistics: mean, sigma and class prior probability"""
    row_num = x_train.shape[0]
    col_num = x_train.shape[1] + 1

    # Combining x_train and y_train
    combined_data = np.concatenate((x_train, y_train), axis=1)

    # Dictionaries have been used to store the mean, standard deviation and class prior probability where the keys
    # are classes
    means = {}
    sigma = {}
    cls_prior_prob = {}

    classes_col = combined_data[:, -1].astype(int)
    for i in np.unique(classes_col):
        tmp = combined_data[np.where(classes_col == i)]
        means[i] = []
        sigma[i] = []
        cls_prior_prob[i] = 0
        cls_prior_prob[i] = ((tmp.shape[0]) / row_num)

        for j in range(col_num - 1):
            means[i].append(np.mean(tmp[:, j]))
            std = np.std(tmp[:, j])
            sigma[i].append(max([std, 1e-6]))

    return means, sigma, cls_prior_prob


def class_probability(obs, mean, sigma, prob):
    """Bayes classifier which returns the probability of class given observation, as well as the maximum probability and
    the related class name"""
    # Step 1: calculating the probability of obs given each class
    obs_conditional_prob = dict()
    for i in range(len(mean)):
        obs_conditional_prob[i] = multivariate_normal.pdf(obs, mean[i], sigma[i])

    # Step 2: calculating the probability of the class given observation
    class_prob = dict()
    tmp_prob = []
    total_obs_prob = 0
    for i in range(len(mean)):
        total_obs_prob += obs_conditional_prob[i] * prob[i]
        tmp_prob.append((obs_conditional_prob[i] * prob[i]))

    for i in range(len(mean)):
        class_prob[i] = tmp_prob[i] / (total_obs_prob + EPS)

    # Finding the class with maximum probability
    max_prob_key = max(class_prob, key=class_prob.get)
    max_prob_value = class_prob[max_prob_key]

    return class_prob, max_prob_key, max_prob_value


def bayes_error(x_test, y_test, x_train, y_train):
    """Creating a comparison table which includes a copy of x_test, y_test, and the predicted class.
    The function returns the error percentage which compares the actual class in y_test with predicted
    class as well as the comparison table"""
    means, sigma, cls_prior_prob = train_stats(x_train, y_train)
    combined_x_y = np.concatenate((x_test, y_test), axis=1)
    predicted_class = []
    test_row_num = x_test.shape[0]

    for row in range(test_row_num):
        class_prob, max_prob_key, max_prob_value = class_probability(x_test[row], means, sigma, cls_prior_prob)
        predicted_class.append(max_prob_key)
    predicted_class = np.reshape(predicted_class, (test_row_num, 1))
    comparison_table = np.concatenate((combined_x_y, predicted_class), axis=1)

    error = 0
    for row in comparison_table:
        if row[-1] != row[-2]:
            error += 1
    error_percentage = error / comparison_table.shape[0]

    # The below is to test the answer using sklearn built-in bayes function. The answers reported on my
    # paper has been validated using the below function
    gnb = GaussianNB()
    gnb.fit(x_train, np.squeeze(y_train))
    y_pred = gnb.predict(x_test)
    True_error = 1 - metrics.accuracy_score(y_test, y_pred)

    return comparison_table, error_percentage, True_error


def main():
    # loading previously prepared data sets: original versions, and boolean versions
    with open('data_sets.pkl', 'rb') as data_sets:
        blackboard_datasets = pickle.load(data_sets)
    part1_data = blackboard_datasets['part1_data']
    part2_data = blackboard_datasets['part2_data']
    part3_data = blackboard_datasets['part3_data']

    part1_data_bool = blackboard_datasets['part1_data_bool']
    part2_data_bool = blackboard_datasets['part2_data_bool']
    part3_data_bool = blackboard_datasets['part3_data_bool']

    # loading previously prepared vote data set
    with open('vote_data_set.pkl', 'rb') as vote_data_sets:
        vote_data_set = pickle.load(vote_data_sets)
    vote_data = vote_data_set['vote_data']

    # Train and test sets
    x_train_1, x_test_1, y_train_1, y_test_1 = data_train_test(part1_data)
    x_train_2, x_test_2, y_train_2, y_test_2 = data_train_test(part2_data)
    x_train_3, x_test_3, y_train_3, y_test_3 = data_train_test(part3_data)
    vote_x_train, vote_x_test, vote_y_train, vote_y_test = data_train_test(vote_data)
    x_train_1_bool, x_test_1_bool, y_train_1_bool, y_test_1_bool = data_train_test(part1_data_bool)
    x_train_2_bool, x_test_2_bool, y_train_2_bool, y_test_2_bool = data_train_test(part2_data_bool)
    x_train_3_bool, x_test_3_bool, y_train_3_bool, y_test_3_bool = data_train_test(part3_data_bool)

    # Writing comparison tables into output files
    comparison_table_1_bool, error_1_bool, True_error_1_bool = bayes_error(x_test_1_bool, y_test_1_bool, x_train_1_bool,
                                                                           y_train_1_bool)
    np.savetxt(
        'output_files/Bayes/comparison_table_part1_bool.csv',
        comparison_table_1_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_1_bool, True_error_1_bool}')

    comparison_table_2_bool, error_2_bool, True_error_2_bool = bayes_error(x_test_2_bool, y_test_2_bool,
                                                                           x_train_2_bool,
                                                                           y_train_2_bool)
    np.savetxt(
        'output_files/Bayes/comparison_table_part2_bool.csv',
        comparison_table_2_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_2_bool, True_error_2_bool}')

    comparison_table_3_bool, error_3_bool, True_error_3_bool = bayes_error(x_test_3_bool, y_test_3_bool, x_train_3_bool,
                                                                           y_train_3_bool)
    np.savetxt(
        'output_files/Bayes/comparison_table_part3_bool.csv',
        comparison_table_3_bool,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_3_bool, True_error_3_bool}')

    comparison_table_1, error_1, True_error_1 = bayes_error(x_test_1, y_test_1, x_train_1, y_train_1)
    np.savetxt(
        'output_files/Bayes/comparison_table_part1.csv', comparison_table_1,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_1, True_error_1}')

    comparison_table_2, error_2, True_error_2 = bayes_error(x_test_2, y_test_2, x_train_2, y_train_2)
    np.savetxt(
        'output_files/Bayes/comparison_table_part2.csv', comparison_table_2,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_2, True_error_2}')

    comparison_table_3, error_3, True_error_3 = bayes_error(x_test_3, y_test_3, x_train_3, y_train_3)
    np.savetxt(
        'output_files/Bayes/comparison_table_part3.csv', comparison_table_3,
        delimiter=',', header='feature1,feature2,class,predicted_class', fmt='%0d',
        footer=f'Error:{error_3, True_error_3}')

    comparison_table_vote, error_vote, True_error_vote = bayes_error(vote_x_test, vote_y_test, vote_x_train,
                                                                     vote_y_train)
    np.savetxt(
        'output_files/Bayes/comparison_table_vote.csv', comparison_table_vote,
        delimiter=',', header='feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,'
                              'feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,'
                              'class,predicted_class', fmt='%0d',
        footer=f'Error:{error_vote, True_error_vote}')


if __name__ == '__main__':
    main()
