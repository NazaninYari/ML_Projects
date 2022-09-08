import numpy as np
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
EPS = 1e-6


np.random.seed(0)


def data_train_test(data_set: np.array):
    """Creating test and train data sets"""
    row_num, col_num = data_set.shape

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
# the related class name"""
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
        class_prob[i] = tmp_prob[i] / (total_obs_prob+ EPS)

    # Finding the class with maximum probability
    max_prob_key = max(class_prob, key=class_prob.get)
    max_prob_value = class_prob[max_prob_key]

    return class_prob, max_prob_key, max_prob_value


def bayes_error(x_test, y_test, x_train, y_train):
    """Creating a comparison table which includes a copy of x_test, y_test, and the predicted class. The function returns
# only the error percentage which compares the actual class in y_test with predicted class"""
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

    # Please uncomment the below to test the answer using sklearn built-in bayes function. The answers reported on my
    # paper has been validated using the below function
    # gnb = GaussianNB()
    # gnb.fit(x_train, y_train)
    # y_pred = gnb.predict(x_test)
    # True_error = 1 - metrics.accuracy_score(y_test, y_pred)

    return error_percentage
