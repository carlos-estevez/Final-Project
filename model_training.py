from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

def create_network(n_vocab):
    """ Create network structure """
    model = Sequential()

    model.add(LSTM(
    500,
    return_sequences=True,
    time_major=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.3)) 
    model.add(LSTM(500))
    model.add(Dense(250))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model

def train(model, network_input, network_output):
    """ Train the model """
    
    filepath = "weights.{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode="min"
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=40, batch_size=64, callbacks=callbacks_list)