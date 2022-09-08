import logging
import pickle
import numpy as np
import sklearn.metrics

EPS = 1e-6
logging.getLogger('').setLevel(logging.INFO)


def euclidean_distance(x1, x2):
    """Calculate Euclidean distance."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cluster_distance(dataset, C1, C2):
    """ Calculating the Euclidean distance between 2 clusters using single linkage (the clusters hold the indices)."""
    single_linked_distance = np.math.inf
    for i in C1:
        for j in C2:
            distance = euclidean_distance(dataset[i, :-1], dataset[j, :-1])
            if distance < single_linked_distance:
                single_linked_distance = distance
    return single_linked_distance


def HAC_final(k, dataset):
    # Removing the last column "class" from the data set
    dataset_no_class = dataset[:, :-1]
    num_obs, num_features = dataset_no_class.shape

    # the row and column indices of the lower triangle matrix
    i_s, j_s = np.tril_indices(n=num_obs, k=-1)
    points_combination_indices = list(zip(i_s, j_s))

    # Initializing the cluster with all data points as clusters
    clusters_dict = {}
    for i in range(num_obs):
        clusters_dict[i] = [i]

    # Initializing the distance dictionary which stores all possible combination of each 2 data points along with
    # the distance
    distance_dict = {}
    for indices in points_combination_indices:
        distance_dict[indices] = euclidean_distance(dataset_no_class[indices[0]], dataset_no_class[indices[1]])

    final_clusters_dict = clusters_dict
    final_distance_dict = distance_dict

    while len(clusters_dict) > k:
        logging.info(f'cluster_dict size: {len(clusters_dict)}')

        # Finding the cluster with minimum distance
        min_key = min(final_distance_dict, key=final_distance_dict.get)
        min_distance = final_distance_dict[min_key]

        # Adding a new cluster we just found with minimum distance to the clusters dictionary
        cluster_max_key = max(k for k, v in final_clusters_dict.items())
        final_clusters_dict[cluster_max_key + 1] = final_clusters_dict[min_key[0]] + final_clusters_dict[min_key[1]]
        logging.info(f'New cluster size: {len(final_clusters_dict[cluster_max_key + 1])}')

        # Removing min_key from the clusters we just merged
        del final_clusters_dict[min_key[0]]
        del final_clusters_dict[min_key[1]]

        # Removing the clusters including min_key[0] and min_key[1] from distance dict to add the updated distances
        tmp_indices = []
        for indices in final_distance_dict.keys():
            if min_key[0] in indices or min_key[1] in indices:
                tmp_indices.append(indices)

        for indices in tmp_indices:
            del final_distance_dict[indices]

        # Adding the updated distances for the clusters including min_key[0] and min_key[1]
        logging.info('Started updating distances')
        cluster_keys = list(final_clusters_dict.keys())
        cluster_keys.remove(cluster_max_key + 1)
        for i in cluster_keys:
            distance_dict[(cluster_max_key + 1, i)] = cluster_distance(dataset_no_class,
                                                                       clusters_dict[cluster_max_key + 1],
                                                                       clusters_dict[i])
        logging.info('Done updating distances')

    # Updating the cluster dictionary keys to 0 through k-1
    class_range = list(range(k))
    cluster_keys = list(clusters_dict.keys())
    zip_keys = list(zip(class_range, cluster_keys))
    for k in zip_keys:
        clusters_dict[k[0]] = clusters_dict.pop(k[1])

    return clusters_dict


def silhouette(dataset, clusters_dict):
    """# The silhouette used to evaluate HAC."""
    dataset = dataset[:, :-1]
    num_obs, num_features = dataset.shape

    # Initializing a matrix of size (num_obs x num_obs)
    distance_matrix = np.zeros((num_obs, num_obs))

    # Creating a matrix (num_obs x num_obs) which stores the distance between any 2 data points
    for i in range(num_obs):
        for j in range(i):
            distance_matrix[i, j] = euclidean_distance(dataset[i, :], dataset[j, :])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Number of classes
    k = len(clusters_dict)

    # Creating a (num_obs x k) matrix class membership mask
    cls_mask_matrix = np.zeros((num_obs, k))
    for c in range(k):
        cluster_size = len(clusters_dict[c])
        for i in clusters_dict[c]:
            cls_mask_matrix[i, c] = 1.0

        # divide by cluster size so a dot product produces a_i's and b_i's
        cls_mask_matrix[:, c] /= cluster_size

    # Multiplying the two matrices to calculate a_i and b_i's in each row of the new matrix (num_obs x k)
    ai_bi_mat = np.matmul(distance_matrix, cls_mask_matrix)

    a_i = np.zeros((num_obs,))
    i_to_cluster = np.zeros((num_obs,), dtype=np.int)
    for c_idx, i_list in clusters_dict.items():
        for i in i_list:
            i_to_cluster[i] = c_idx

    cluster_size_1_mask = np.ones((num_obs,))
    for i in range(num_obs):
        c = i_to_cluster[i]
        cluster_size = len(clusters_dict[c])
        a_i[i] = ai_bi_mat[i, c] * cluster_size / (cluster_size - 1 + EPS)
        ai_bi_mat[i, c] = np.inf
        if cluster_size == 1:
            cluster_size_1_mask[i] = 0

    b_i = np.amin(ai_bi_mat, axis=1)
    sil_s = (b_i - a_i) / np.maximum(a_i, b_i)
    sil_s *= cluster_size_1_mask
    sil = np.mean(sil_s)

    true_sil = sklearn.metrics.silhouette_score(X=distance_matrix, labels=i_to_cluster, metric="precomputed")
    return sil, true_sil


def main():
    with open('proj2_dataset.pkl', 'rb') as proj2_dataset:
        proj2_dataset = pickle.load(proj2_dataset)
    synthetic_data = proj2_dataset['synthetic_data']
    spambase_data = proj2_dataset['spambase_data']
    synthetic_4000 = proj2_dataset['synthetic_4000']
    standardized_synthetic = proj2_dataset['standardized_synthetic']
    standardized_spambase = proj2_dataset['standardized_spambase']
    standardized_synthetic_4000 = proj2_dataset['standardized_synthetic_4000']

    logging.info('Starting clustering')
    clusters_dict = HAC_final(2, standardized_synthetic_4000[:, [1, 4, -1]])
    logging.info('Done clustering')
    print(silhouette(standardized_synthetic_4000[:, [1, 4, -1]], clusters_dict))


if __name__ == '__main__':
    main()
