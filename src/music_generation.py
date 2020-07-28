import pickle
import numpy
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from src.data_preparation import datatonumber, sequence, normalize

pitchnames = sorted(set(item for item in notes))
notes = 
n_vocab = len(set(notes))


datatonumber(pitchnames)
sequence(notes)
normalize(n_vocab)


def new_network(n_vocab):
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

    model.load_weights("  ")

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network """
    
    first_note = numpy.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[first_note]
    new_song = []

    # Generate 300 notes
    for note in range(300):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction) 
        new_note = int_to_note[index] # Highest probability note
        new_song.append(new_note)

        # Define new pattern by adding new note and eliminate first
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return new_song


def create_midi(new_song):
    """ Converts the output back to notes and create a midi file """
    
    offset = 0
    output_notes = []

    for pattern in new_song:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.SnareDrum()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.SnareDrum()
            output_notes.append(new_note)

        # Increase offset every loop so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp = "test_output.mid" )
