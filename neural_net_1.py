import keras
import numpy as np
from keras.layers import Dense
from sklearn import preprocessing
import pickle
import itertools


def neural_init():
    #  Cria um array de -1 a 1 espaçado 400 vezes iguais
    x1 = np.linspace(-1, 1, 400)
    x2 = np.linspace(-1, 1, 400)

    x = np.array(list(itertools.product(x1, x2)))

    # Função descrita pela rede neural
    y = np.sin(2 * x[:, 0]) + np.cos(x[:, 1])

    #  Média e Desvio Padrão da entrada
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)

    #  Média e Desvio Padrão da saída
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)

    parameters = {'x': x, 'y': y,
                  'mean_x': mean_x, 'mean_y': mean_y,
                  'std_x': std_x, 'std_y': std_y}

    return parameters


def model():

    parameters = neural_init()
    x = parameters['x']
    y = parameters['y']

    mean_x = parameters["mean_x"]
    print("mean_x =", mean_x)

    std_x = parameters['std_x']
    print("std_x =", std_x)

    mean_y = parameters['mean_y']
    print("mean_y =", mean_y)

    std_y = parameters['std_y']
    print("std_y =", std_y)

    #  Normalização da entrada e da saída
    x_norm = (x - mean_x) / std_x
    y_norm = (y - mean_y) / std_y

    #  Camadas da Rede Neural ReLu
    model = keras.models.Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(x_norm, y_norm, epochs=20)  # nb_epoch=20

    res = model.evaluate(x_norm, y_norm)

    print(res)

    print("mean_x =", mean_x)
    print("std_x =", std_x)

    print("mean_y =", mean_y)
    print("std_y =", std_y)

    weights = list(map(lambda layer: layer.get_weights(), model.layers))
    with open('model.pickle', 'wb') as f:
        pickle.dump(weights, f)

    # Ponto esperado para o máximo global
    z1 = [0.7854]
    z2 = [0]
    z = np.vstack([z1, z2]).T
    # Desnormaliza os valores
    z = (z - mean_x) / std_x

    y_pred = model.predict(z)
    print(y_pred)

    # print(y.T)


if __name__ == '__main__':
    model()
