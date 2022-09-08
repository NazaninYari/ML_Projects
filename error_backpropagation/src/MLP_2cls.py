import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Project4.read_data_p4 import read_cancer

EPS = 1e-6
np.random.seed(0)


def weights_network(inputs_size, num_hidden_layers, num_hidden_nodes: []):
    """Initializing a network with random weights
    num_hidden_nodes is a list with each element as the number of nodes with index associated with the hidden layer"""
    # Initializing network to store the hidden and output layers with initial random weights
    weight_network = []

    # We added +1 for the number of rows for bias or W0.
    for hidden_layer in list(range(num_hidden_layers + 1)):
        if hidden_layer == 0:
            weight_network.append(
                {'hidden_layer_' + str(hidden_layer) + str('_weights'): np.random.randn(inputs_size + 1,
                                                                                        num_hidden_nodes[0]) / 100})
        elif hidden_layer == num_hidden_layers:
            weight_network.append(
                {'output_layer' + str('_weights'): np.random.randn(num_hidden_nodes[hidden_layer - 1] + 1,
                                                                   1) / 100})
        else:
            weight_network.append({'hidden_layer_' + str(hidden_layer) + str('_weights'): np.random.randn(
                num_hidden_nodes[hidden_layer - 1] + 1,
                num_hidden_nodes[hidden_layer]) / 100})

    return weight_network


def sigmoid(i):
    """Activation function for hidden layers"""
    return 1 / (1 + np.exp(-i))


def weighted_sum(inputs, weights):
    """Calculate the dot product of input row and weights"""
    bias = weights[-1, :]
    # Excluding class column from inputs
    y = np.matmul(inputs[:, :-1], weights[:-1, :]) + bias
    return y


def forward_propagate(weight_network, row: np.ndarray):
    """Calculating output z's given an input row"""
    inputs = row
    output_network = []
    for layers_idx in range(len(weight_network)):
        output = []
        for hidden_layer, weights in weight_network[layers_idx].items():
            y = weighted_sum(inputs, weights)
            for i in y[0]:
                output.append(sigmoid(i))
            output_network.append({hidden_layer[:-8] + str('_output'): np.expand_dims(np.array(output), axis=1)})
            new_input = list.copy(output)
            # since input row has a class column that we remove when calculating weighted sum, we need to add a place
            # holder for the new inputs
            new_input.append(0)
            inputs = np.expand_dims(np.array(new_input), axis=0)

    return output_network


def backpropagate_errors(weight_network, output_network, row: np.ndarray):
    # last element in a row is the class number
    target = row[:, -1][0]

    # making a copy of the layers_output and deleting the last item to store only hidden layers
    hidden_layers_output = list.copy(output_network)
    del (hidden_layers_output[-1])

    delta = []

    output_nodes_delta = []
    s_i = list(output_network[-1].values())[0][0, 0]
    t_i = target
    output_nodes_delta.append((t_i - s_i) * s_i * (1.0 - s_i))
    output_nodes_delta = np.expand_dims(np.array(output_nodes_delta).T, axis=1)

    delta.append({"output_layer_delta": output_nodes_delta})

    weight_network_reversed = list.copy(weight_network)
    weight_network_reversed = list(reversed(weight_network_reversed))
    hidden_layers_output_reversed = list.copy(hidden_layers_output)
    hidden_layers_output_reversed = list(reversed(hidden_layers_output_reversed))
    hidden_layers_derivatives = derivative_z_network(hidden_layers_output_reversed)

    # Constructing hidden layers deltas
    num_hidden_layers = len(hidden_layers_derivatives)
    for i in range(num_hidden_layers):
        weight_mul_error = np.matmul(list(weight_network_reversed[i].values())[0][:-1, :], list(delta[-1].values())[0])
        error = np.multiply(list(hidden_layers_derivatives[i].values())[0], weight_mul_error)
        delta.append({'hidden_layer_' + str(num_hidden_layers - 1 - i) + str('_delta'): error})
    return list(reversed(delta))


def derivative_z_network(hidden_layers_output_reversed):
    """Calculating the derivatives of z's and creating a network"""
    hidden_layer_derivatives = []
    for layer in hidden_layers_output_reversed:
        for key, value in layer.items():
            deriv_list = []
            for z in value.T[0]:
                deriv_list.append(sigmoid_derivative(z))
            deriv_list = np.expand_dims(np.array(deriv_list), axis=1)
            hidden_layer_derivatives.append({key + str('_derivative'): deriv_list})
    return hidden_layer_derivatives


def sigmoid_derivative(i):
    """Calculating derivative of sigmoid function"""
    return i * (1 - i)


def update_weight(weight_network, output_network, delta, learning_rate, row: np.ndarray):
    weight_delta_no_momentum = []
    new_weight_no_momentum = []
    num_hidden_layers = len(weight_network)

    # Adding [1] to all the x's to update the bias weights
    bias_x = np.expand_dims(np.array([1]), axis=1)
    for i in range(num_hidden_layers):
        key = list(weight_network[i].keys())[0]
        if i == 0:
            delta_mul_x = np.matmul(np.concatenate(((row[:, :-1]).T, bias_x), axis=0), list(delta[i].values())[0].T)
        else:
            delta_mul_x = np.matmul(np.concatenate((list(output_network[i - 1].values())[0], bias_x), axis=0),
                                    list(delta[i].values())[0].T)
        w_delta = learning_rate * delta_mul_x
        updated_weight = w_delta + list(weight_network[i].values())[0]

        weight_delta_no_momentum.append({key + str('_delta'): w_delta})
        new_weight_no_momentum.append({key + str('_updated'): updated_weight})
    # weight_delta_no_momentum
    return new_weight_no_momentum


def train_network(weight_network, dataset, learning_rate, epoch_num):
    """Training network for a fixed number of epochs"""
    n_output_nodes = 1
    row_num, _ = dataset.shape

    row_errors = []
    epoch_errors = []
    updated_weight_network = []
    for epoch in range(epoch_num):
        epoch_error = 0
        prediction = []
        for row in dataset:
            row = np.expand_dims(row, axis=0)
            output_network = forward_propagate(weight_network, row)
            # last item in the output network is the output_layer's output
            output_nodes = list(output_network[-1].values())[0].T[0][0]
            if output_nodes >= 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
            target_output = int(row[0, -1])
            # compute error sum of squared error
            row_error = (target_output - output_nodes) ** 2
            row_errors.append(row_error)
            epoch_error += row_error
            print(f'{row_error=}')
            delta = backpropagate_errors(weight_network, output_network, row)
            weight_network = update_weight(weight_network, output_network, delta, learning_rate, row)
            updated_weight_network = weight_network
        # average error for each epoch
        epoch_error /= row_num
        epoch_errors.append(epoch_error)

        # compute prediction error for training set to see how well it fit the data
        correct_predict = np.equal(np.array(prediction), (dataset[:, -1]))
        prediction_error = 1 - np.sum(correct_predict) / dataset.shape[0]

        print(f'{epoch=}, {epoch_error=}')
    print(prediction_error)

    plt.plot(epoch_errors, label='Epoch Error')
    plt.xlabel('epoch number')
    plt.ylabel('epoch error')
    plt.title('Epoch Error _ Learning Rate: 0.1 _ 2 Hidden Layers')
    plt.legend()
    plt.show()

    return updated_weight_network


def predict(updated_weight_network, x_test, y_test):
    # all the function accepts the dataset with the last column being the class column. Therefore, we concatenate x, y
    test_data = np.concatenate((x_test, y_test), axis=1)
    predicted_class = []
    for row in test_data:
        row = np.expand_dims(row, axis=0)
        # in the output network from forward prop, we get a list in which the last element is the output node dictionary
        output = list(forward_propagate(updated_weight_network, row)[-1].values())[0][0][0]
        if output >= 0.5:
            predicted_class.append(1)
        else:
            predicted_class.append(0)

    predicted_class = np.expand_dims(np.array(predicted_class), axis=1)
    correct_predict = np.equal(np.array(predicted_class), y_test)
    prediction_error = 1 - np.sum(correct_predict) / x_test.shape[0]
    return predicted_class, prediction_error


def data_train_validation_test(data_set):
    """Creating test, validation and train data sets"""
    # Shuffle the data
    np.random.shuffle(data_set)

    # Including all rows and columns except the last column (class)
    x = data_set[:, :-1]

    # vector of classes (last column)
    y = data_set[:, -1]

    y = np.expand_dims(y, axis=1)

    # Selecting 90% of observation as training data and validation, and 10% as test data
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(x, y, test_size=0.10)

    # Selecting 60% of dataset as training and 30% as validation set or 2/3 of the previously selected 90% of data and
    # 1/3 of the previously selected 90% of data
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_validation, y_train_validation,
                                                                    test_size=1 / 3)

    return x_train, x_test, x_validation, y_train, y_test, y_validation


def main():
    cancer_data = read_cancer()
    x_train, x_test, x_validation, y_train, y_test, y_validation = data_train_validation_test(cancer_data)

    x_train_cancer_rows, x_train_cancer_features = x_train.shape

    # for testing 0 number of hidden layer, please choose 1 for num_hidden_layers, and [0] for num_hidden_nodes
    network_weight_cancer = weights_network(inputs_size=x_train_cancer_features, num_hidden_layers=2,
                                            num_hidden_nodes=[5, 5])

    cancer_train_data = np.concatenate((x_train, y_train), axis=1)

    updated_weight = train_network(network_weight_cancer, cancer_train_data, 0.1, 20)
    # predicted_class, predicted_error = predict(updated_weight, x_validation, y_validation)
    predicted_class, predicted_error = predict(updated_weight, x_test, y_test)
    print("\n predicted error is: ")
    print(predicted_error)


if __name__ == '__main__':
    main()
