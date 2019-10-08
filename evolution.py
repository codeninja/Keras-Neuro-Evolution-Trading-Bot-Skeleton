from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.Population import Population

import pandas as pd
import numpy as np
from time import time


def build_model():
    # TODO: num_inputs should count the total number of inputs from the dataframe
    num_inputs = 11
    # TODO: this should evolve with varying generations
    hidden_nodes = num_inputs * 4
    num_outputs = 3


    # TODO: The model should evolve to an LSTM
    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')

    # # Sequential model for time series data.
    # model = Sequential()
    #
    # # First layer, 50 neurons, return sequence true as we'll  have other layers.
    # # Input shape (number of timesteps, number of indicators)
    # model.add(LSTM(units=13, return_sequences=True, input_shape=(x_t.shape[1], 13)))
    #
    # # dropout to prevent overfitting
    # model.add(Dropout(0.2))
    #
    # # Create several more LSTM and Dropout layers, this will allow the LSTM to gain a higher level understanding of the relationship of attributes.
    # model.add(LSTM(units=100, activation='relu', return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=100, activation='softmax'))
    # model.add(Dropout(0.2))
    #
    # # model.add(LSTM(units=13))
    # # model.add(Dropout(0.2))
    #
    # # Add a dense layer to predict the value
    # model.add(Dense(units = 1, activation='linear'))
    # model.compile(optimizer = 'adam', loss = 'mse', metrics=['mse'])
    return model

if __name__ == '__main__':
    # suppress tf GPU logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    pop_size = 50
    mutation_rate = 0.05
    mutation_scale = 0.3
    starting_cash = 100.
    trading_fee = 0.005
    generations = 2

    # generate random test data
    test_size = 300
    # np.random.seed(42)
    # prices = np.random.normal(10, 0.1, test_size)
    # inputs = np.random.rand(test_size, 4) * 2 - 1

    # TODO: pull a subset of the last 90 days.
    df_raw = pd.read_csv("./data/training/btc_preprocessed.csv", index_col=[0])

    print('Len DF_Raw', len(df_raw), df_raw.shape, df_raw.columns)
    columns = ['open','close','high','low','volume', 'MACD', 'SIGNAL', '41 period SMA', 'BB_UPPER','BB_MIDDLE','BB_LOWER']

    # define the data window we want to segragate from the full dataset.
    # since our data is in 1 hour timesteps we want to go back a certain number of days
    timesteps_per_day = 6
    train_days = (365 * 2) * timesteps_per_day
    test_days = 365 * timesteps_per_day

    # split df into training and testing
    training_window = len(df_raw) - train_days
    testing_window = len(df_raw) - test_days

    print('training window', train_days, test_days, training_window, testing_window)

    dft = df_raw[training_window:testing_window]
    dfv = df_raw[testing_window:]

    # training datafraes
    dft_prices = dft['close']
    dft_inputs = dft[columns]

    # validation datafraes
    dfv_prices = dfv['close']
    dfv_inputs = dfv[columns]

    # build initial population
    pop = Population(pop_size, build_model, mutation_rate,
                     mutation_scale, starting_cash, dft_prices[0], trading_fee, )

    # run defined number of evolutions
    for i in range(generations):
        start = time()
        pop.evolve(dft_inputs, dft_prices)
        print('\n\nDuration: {0:.2f}s'.format(time()-start))

    pop.validate(dfv_inputs, dfv_prices)
