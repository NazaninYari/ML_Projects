import math
import pickle
import numpy as np


def Euclidean_distance(X1, X2):
    """Method to calculate Euclidean distance to the power of 2"""
    return np.sum((X1 - X2) ** 2)


def initial_centroids(k, dataset):
    """Method to select the initial k centroids based on K-means++ instead of randomly choosing the initial centroids"""
    num_obs, num_features = dataset.shape
    centroids = []

    # Assigning a random row as the first centroid
    random_row_num = np.random.randint(0, num_obs)
    centroids.append(list(dataset[random_row_num, :-1]))

    # Computing the rest of the centroids (k-1)
    for i in range(k - 1):

        # Initializing a list to store distances of data points from the nearest centroid
        distances = []

        for j in range(num_obs):
            data_point = dataset[j, :-1]
            min_distance = math.inf

            # Calculating the distance of the data point from each of the centroids and store the minimum distance
            for c in range(len(centroids)):
                tmp_distance = Euclidean_distance(data_point, centroids[c])
                min_distance = min(min_distance, tmp_distance)
            distances.append(min_distance)

        distances = np.array(distances)

        # Selecting data point with maximum distance as our next centroid
        next_centroid = dataset[np.argmax(distances), :-1]
        centroids.append(list(next_centroid))

    centroids = np.reshape(centroids, (k, num_features - 1))

    return centroids


def assign_clusters(centroids, dataset):
    """Assigning clusters based on closest centroid -> vector with size num_obs x 1"""
    clusters = []
    num_obs, num_features = dataset.shape

    # Computing the distance to each centroid for each data point
    for i in range(num_obs):
        distances = []
        for centroid in centroids:
            distances.append(Euclidean_distance(centroid, dataset[i, :-1]))

        # Adding the index of the data points to the nearest clusters
        min_distance = min(distances)
        for cls, distance in enumerate(distances):
            if distance == min_distance:
                clusters.append(cls)
                break

    clusters = np.array(clusters).astype(int)
    clusters = np.reshape(clusters, (num_obs, 1))

    return clusters


def compute_centroids(k, clusters, dataset):
    """Computing new centroids based on each cluster's mean"""
    num_obs, num_features = dataset.shape
    new_centroids = []
    combined_data = np.concatenate((dataset[:, :-1], clusters), axis=1)
    unique_clusters = list(range(k))

    for cluster in unique_clusters:
        cluster_data_subset = combined_data[np.where(combined_data[:, -1] == cluster)]
        if cluster_data_subset.size > 0:
            new_centroids.append(np.mean(cluster_data_subset[:, :-1], axis=0))

        # Choosing random data point if a cluster does not have any data associated with it
        else:
            random_row_num = np.random.randint(0, num_obs)
            new_centroids.append(combined_data[random_row_num, :-1])

    new_centroids = np.array(new_centroids)
    new_centroids = np.reshape(new_centroids, (k, num_features - 1))

    return new_centroids


def final_clusters(k, dataset):
    """Computing final clusters where the centroids do not change anymore"""
    # Step 1: select initial centroids
    initial_centroid = initial_centroids(k, dataset)

    # Step 2: Assign clusters to each data point based on closest distance to cluster's centroid
    initial_cluster = assign_clusters(initial_centroid, dataset)

    iteration_num = 1

    final_centroid = initial_centroid
    final_cluster = initial_cluster

    convergence = False
    while not convergence:
        # Calculating the next iteration centroid
        next_iteration_centroid = compute_centroids(k, final_cluster, dataset)

        if np.array_equal(next_iteration_centroid, final_centroid):
            convergence = True
            break
        else:
            next_iteration_cluster = assign_clusters(next_iteration_centroid, dataset)

        final_centroid = next_iteration_centroid
        final_cluster = next_iteration_cluster

        iteration_num += 1
    return final_cluster, final_centroid, iteration_num


def k_mean(k, dataset):
    """Adding the new clusters to the data in the last column"""
    final_cluster, final_centroid, iteration_num = final_clusters(k, dataset)
    combined_data = np.concatenate((dataset, final_cluster), axis=1)
    final_combined_data = []

    for cluster in range(k):
        frequencies = []

        # Creating a subset of data where the last column is equal to the cluster in range of k
        cluster_data_subset = combined_data[np.where(combined_data[:, -1] == cluster)]

        # Counting the frequency of each class in each subset dataset to assign the class number associated with the
        # majority count to cluster
        (class_num, counts) = np.unique(cluster_data_subset[:, -2], return_counts=True)

        class_num_int = class_num[0]
        counts_int = counts[0]

        frequencies.append([class_num_int, counts_int])
        frequencies = np.reshape(np.array(frequencies), (len(frequencies), 2))

        # Finding the index of the highest frequency
        cluster_idx = np.argmax(frequencies[:, -1], axis=0)

        # Finding the final cluster number which is the first item in the tuple where the second item is the maximum
        cluster_num = frequencies[cluster_idx.astype(int)][0]

        obs_num = cluster_data_subset.shape[0]

        # Creating the column with the cluster number to be added to the dataset
        cluster_column = np.reshape(np.repeat(cluster_num, obs_num), (obs_num, 1))

        # Creating a new subset with the cluster number in the last column
        final_cluster_data_subset = np.concatenate((cluster_data_subset[:, :-1], cluster_column), axis=1)

        final_combined_data.append(final_cluster_data_subset)

    # Converting the final dataset into a numpy matrix
    final_combined_data = np.concatenate(final_combined_data, axis=0)

    return final_combined_data


def k_mean_error(final_combined_data):
    """Calculating the K-means error based on the actual classes"""
    error = 0

    for row in final_combined_data:
        if row[-1] != row[-2]:
            error += 1
    error_percentage = error / final_combined_data.shape[0]
    return error_percentage


def main():
    with open('proj2_dataset.pkl', 'rb') as proj2_dataset:
        proj2_dataset = pickle.load(proj2_dataset)
    spambase_data = proj2_dataset['spambase_data']
    synthetic_data = proj2_dataset['synthetic_data']

    final_combined_data = k_mean(2, spambase_data[:, [20, 29, 51, 15, 5, 23, 32]])
    print(k_mean_error(final_combined_data))


if __name__ == '__main__':
    main()
