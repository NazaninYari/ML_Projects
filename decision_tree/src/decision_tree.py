import copy
import logging
import pprint
import random

import numpy as np
from sklearn.model_selection import train_test_split

from Project3.read_data import split_to_x_y, read_data

EPS = 1e-6
np.random.seed(400)
random.seed(0)
logging.getLogger('').setLevel(logging.INFO)


def dataset_entropy(x_train, y_train):
    """Calculating the entropy of the given dataset"""
    entropy_S = 0

    # Unique values of the class column (y_train)
    cls = np.unique(y_train)

    # total number of data points
    row_num = x_train.shape[0]

    for value in cls:
        prob = np.count_nonzero(y_train == value) / row_num
        entropy_S += -prob * np.log2(prob)
    return entropy_S


def calculate_feature_entropy(x_train, y_train, feature_num):
    """Calculating the entropy of a given feature"""

    # Unique values of the class column (y_train)
    cls_values = np.unique(y_train)

    # Unique values of the feature column
    feature_values = np.unique(x_train[:, feature_num])

    # total number of data points
    row_num = x_train.shape[0]

    # Initializing the entropy of a specific feature (column)
    feature_entropy = 0
    for value in feature_values:
        # Initializing the entropy of a specific value of the feature
        feature_values_entropy = 0

        data_subset = x_train[np.where(x_train[:, feature_num] == value)]
        data_subset_rows = data_subset.shape[0]

        for cls in cls_values:
            num = np.count_nonzero(y_train == cls)
            den = data_subset_rows
            prob = num / (den + EPS)
            feature_values_entropy += -prob * np.log2(prob + EPS)
        prob2 = den / row_num
        feature_entropy += -prob2 * feature_values_entropy

    return feature_entropy


def info_gain(dataset_entropy, feature_entropy):
    """Calculating the information gain for specific feature"""
    return dataset_entropy + feature_entropy


def best_ig(x_train, y_train):
    """Finding the feature with the highest information gain"""

    # dataset entropy
    entropy_s = dataset_entropy(x_train, y_train)

    IG_list = []

    # Unique values of the class column
    cls_values = np.unique(y_train)

    # list of the indices of the dataset features
    feature_list = list(range(0, x_train.shape[1]))

    for feature in feature_list:
        feature_entropy = calculate_feature_entropy(x_train, y_train, feature)
        IG_list.append(info_gain(entropy_s, feature_entropy))

    # Finding the index of the feature with maximum gain
    max_ig = 0
    for idx, ig in list(enumerate(IG_list)):
        if ig > max_ig:
            max_ig = ig
            max_idx = idx

    return max_idx, max_ig


def data_subset(x, y, feature_num, value):
    """Creating a subset of dataset filtered for a specific value of a feature excluding the feature column"""
    # Creating combined dataset of x_train and y_train
    combined_dataset = np.concatenate((x, y), axis=1)
    rows = np.where(combined_dataset[:, feature_num] == value)
    sub_x = x[rows]
    sub_y = y[rows]
    sub_x = np.delete(sub_x, feature_num, axis=1)
    return sub_x, sub_y


def decision_tree(x_train, y_train, feature_names, y_size):
    """Building the decision tree"""
    # Finding the best_feature_idx or the feature number with maximum info gain
    best_feature_idx = best_ig(x_train, y_train)[0]
    # the index of all features except the best feature (with maximum info gain)
    remaining_features = tuple(feature_names[:best_feature_idx] + feature_names[best_feature_idx + 1:])

    # Unique values of the selected best_feature_idx
    feature_values = np.unique(x_train[:, best_feature_idx])
    feature_name = feature_names[best_feature_idx]
    value_dict = {}
    tree = [feature_name, value_dict]

    for value in feature_values:

        # Creating a data subset with the unique values of the selected feature
        sub_x_train, sub_y_train = data_subset(x_train, y_train, best_feature_idx, value)

        # Unique values of class column
        unique_ys, counts = np.unique(sub_y_train, return_counts=True)
        # Compute y distribution which stores the counts of each class label
        y_distrib = np.zeros(shape=(y_size,), dtype=np.int)
        for y_idx, count in zip(unique_ys, counts):
            y_distrib[y_idx] = count
        y_distrib = tuple(y_distrib)

        if unique_ys.size == 1:
            node_val = unique_ys[0]
        # if no more features left, pick the most frequent class
        elif len(remaining_features) == 0:
            node_val = unique_ys[np.argmax(counts)]
        else:
            node_val = decision_tree(sub_x_train, sub_y_train, remaining_features, y_size)
        value_dict[value] = [node_val, y_distrib]

    return tree


def predict_row(row, tree: list, feature_index: dict):
    """prediction of class for a data point"""
    feature_name, value_dict = tree

    feature_value = row[feature_index[feature_name]]
    if feature_value in value_dict:
        node_value, _ = value_dict[feature_value]
    else:
        node_value = np.argmax(sum_y_distribution(value_dict))

    if isinstance(node_value, list):
        return predict_row(row, node_value, feature_index)
    else:
        return node_value


def sum_y_distribution(value_dict):
    # number of unique classes is equal to length of y_distrib
    num_ys = len(list(value_dict.values())[0][1])
    sum_y_distribs = np.zeros(shape=(num_ys,))
    for _, y_distrib in value_dict.values():
        sum_y_distribs += y_distrib
    return sum_y_distribs


def predict_batch(x_test, tree, feature_names):
    """Returning an array which stores the predicted classification for the test dataset"""
    # getting index of the feature names list
    feature_index = {feat: i for i, feat in enumerate(feature_names)}
    y_predicted = []
    for row in x_test:
        y_predicted.append(predict_row(row, tree, feature_index))

    y_predicted = np.array(y_predicted)
    y_predicted = np.expand_dims(y_predicted, axis=1)
    return y_predicted


def error_rate(y_predicted: np.array, y_true: np.array):
    """Calculating the error rate of the prediction"""
    diff = np.not_equal(y_predicted, y_true)
    data_size = y_true.shape[0]
    return np.sum(diff) / data_size


def data_train_validation_test(x: np.array, y: np.array):
    """Creating test, validation and train data sets"""
    # Selecting 90% of observation as training data and validation, and 10% as test data
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(x, y, test_size=0.10)

    # Selecting 60% of dataset as training and 30% as validation set or 2/3 of the previously selected 90% of data and
    # 1/3 of the previously selected 90% of data
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_validation, y_train_validation,
                                                                    test_size=1 / 3)

    return x_train, x_test, x_validation, y_train, y_test, y_validation


def compute_feature_depths(tree_y_distrib: list, feature_with_depths: list, depth):
    """Computing the depth of the features in the tree"""
    tree, _ = tree_y_distrib
    feature_name, value_dict = tree

    feature_with_depths.append([depth, tree_y_distrib])

    for node_y_distrib in value_dict.values():
        node, _ = node_y_distrib
        if isinstance(node, list):
            compute_feature_depths(node_y_distrib, feature_with_depths, depth + 1)


def prune_single_feature(sorted_features: list, pick_randomly=False):
    """Pruning the tree on a single subtree"""
    # Finding the feature with the highest depth in the tree
    if pick_randomly:
        i = random.choices(range(len(sorted_features)), weights=[v[0] for v in sorted_features])[0]
    else:
        i = 0
    # find the feature with maximum depth and remove the feature from feature_depth
    feature_to_prune = sorted_features.pop(i)
    # Finding the index of the class with the majority counts
    _, node_y_distrib = feature_to_prune
    node, y_distrib = node_y_distrib
    feature_name, _ = node
    logging.info(f'Pruning: {feature_name}')
    majority_y = np.argmax(y_distrib)
    # Replacing the subtree with the class with majority counts
    node_y_distrib[0] = majority_y


def prune(tree, x_validation, y_validation, feature_names):
    """Pruning the tree based on reduced error pruning and returning the pruned tree"""
    feature_with_depths = []
    compute_feature_depths([tree, None], feature_with_depths, 0)
    sorted_features = sorted(feature_with_depths, key=lambda v: v[0], reverse=True)
    print([(v[1][0][0], v[0]) for v in sorted_features])

    prediction_tree = predict_batch(x_validation, tree, feature_names)
    error_tree = error_rate(prediction_tree, y_validation)
    logging.info(f'Full tree error: {error_tree}')

    # Continue while the feature_depth is non empty. When a feature is pruned it's removed from feature_depth too.
    while len(sorted_features) > 1:
        tree_copy = copy.deepcopy(tree)
        prune_single_feature(sorted_features)

        prediction_pruned_tree = predict_batch(x_validation, tree, feature_names)
        error_pruned_tree = error_rate(prediction_pruned_tree, y_validation)
        logging.info(f'Pruned tree error: {error_pruned_tree}')

        if error_pruned_tree <= error_tree:
            error_tree = error_pruned_tree
        else:
            logging.info('Stopping the pruning')
            tree = tree_copy
            break

    print([(v[1][0][0], v[0]) for v in sorted_features])
    return tree, error_tree


def testdata1():
    feature_names = ['Outlook', 'Temp', 'Humidity', 'Wind']
    test_data = [['sunny', 'hot', 'high', 'weak', 'no'],
                 ['sunny', 'hot', 'high', 'strong', 'no'],
                 ['overcast', 'hot', 'high', 'weak', 'yes'],
                 ['rain', 'mild', 'high', 'weak', 'yes'],
                 ['rain', 'cool', 'normal', 'weak', 'yes'],
                 ['rain', 'cool', 'normal', 'strong', 'no'],
                 ['overcast', 'cool', 'normal', 'strong', 'yes'],
                 ['sunny', 'mild', 'high', 'weak', 'no'],
                 ['sunny', 'cool', 'normal', 'weak', 'yes'],
                 ['rain', 'mild', 'normal', 'weak', 'yes'],
                 ['sunny', 'mild', 'normal', 'strong', 'yes'],
                 ['overcast', 'mild', 'high', 'strong', 'yes'],
                 ['overcast', 'hot', 'normal', 'weak', 'yes'],
                 ['rain', 'mild', 'high', 'strong', 'no']]
    return np.array(test_data), feature_names


def testdata2():
    test_data = [['none', '0 to $15K', 'high', 'bad', 'high'], ['none', '$15K to $35K', 'high', 'unknown', 'high'],
                 ['none', '$15K to $35K', 'low', 'unknown', 'moderate'],
                 ['none', '0 to $15K', 'low', 'unknown', 'high'],
                 ['none', 'over $35K', 'low', 'unknown', 'low'], ['adequate', 'over $35K', 'low', 'unknown', 'low'],
                 ['none', '0 to $15K', 'low', 'bad', 'high'], ['adequate', 'over $35K', 'low', 'bad', 'moderate'],
                 ['none', 'over $35K', 'low', 'good', 'low'], ['adequate', 'over $35K', 'high', 'good', 'low'],
                 ['none', '0 to $15K', 'high', 'good', 'high'], ['none', '$15K to $35K', 'high', 'good', 'moderate'],
                 ['none', 'over $35K', 'high', 'good', 'low'], ['none', '$15K to $35K', 'high', 'bad', 'high']]
    feature_names = ['collateral', 'income', 'debt', 'credit history', 'risk']
    return np.array(test_data), feature_names


def car_data():
    dataset = read_data("/Users/nazanin/Documents/Machine Learning/Project3_NazaninYari/Input/car.csv")
    feature_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    return dataset, feature_names


def main():
    dataset, feature_names = car_data()
    x, y, y_id_to_label = split_to_x_y(dataset)
    x_train, x_test, x_validation, y_train, y_test, y_validation = data_train_validation_test(x, y)

    tree = decision_tree(x_train, y_train, feature_names, y_size=len(y_id_to_label))
    print("Tree: \n")
    pprint.pprint(tree)

    y_pred = predict_batch(x_test, tree, feature_names)
    error = error_rate(y_pred, y_test)
    print(f'\nerror on test: {error}')

    prune(tree, x_validation, y_validation, feature_names)
    print("Pruned Tree: \n")
    pprint.pprint(tree)

    y_pred = predict_batch(x_test, tree, feature_names)
    pruned_error = error_rate(y_pred, y_test)
    print(f'\nerror on test with pruned tree: {pruned_error} ({pruned_error - error})')


if __name__ == '__main__':
    main()
