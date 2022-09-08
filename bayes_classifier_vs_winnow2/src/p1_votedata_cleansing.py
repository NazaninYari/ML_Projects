import csv
import pickle
import pprint

import numpy as np

from p1_data_to_boolean import write_data, data_stats

# Class names -> numbers
cls_dict = {'republican': 0, 'democrat': 1}


def read_votedata(path):
    """Reading data and changing 'y' to 1, 'n' to 0 and the missing values to randomly selected 1 or 0. The class names
    also have been changed to {republican, democrat} -> {0, 1} and moved to the last column"""
    with open(path, 'r') as dataset:
        csv_reader = csv.reader(dataset)
        data_set = []

        for row in csv_reader:
            tmp_row = []
            for value in row[1:]:
                if value == 'y':
                    tmp_row.append(1)
                elif value == 'n':
                    tmp_row.append(0)
                elif value == '?':
                    np.random.seed(0)
                    tmp_row.append(np.random.randint(2))

            tmp_row.append(cls_dict[row[0]])
            data_set.append(tmp_row)

    return np.array(data_set)


def main():
    vote_data = read_votedata('input_files/house-votes-84.csv')

    # Writing the new converted data list into a file
    write_data('input_files/boolean_files/bool_votedata.csv', vote_data)

    vote_cov, vote_means, vote_std = data_stats(vote_data)

    # dumping dataset into output to be used in the Winnow2 and Bayes programs
    vote_data_set = {
        'vote_data': vote_data,
    }

    with open('vote_data_set.pkl', 'wb') as output:
        pickle.dump(vote_data_set, output)

    print("Covariance is:")
    pprint.pprint(vote_cov)
    print('\nMean is: ')
    pprint.pprint(vote_means)
    print('\nSandard Deviation is: ')
    pprint.pprint(vote_std)


if __name__ == '__main__':
    main()
