import glob
import pickle
import numpy
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def generate():
    """ Generates the midi file """

    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

   
    pitchnames = sorted(set(item for item in notes))

    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
    
def sequence(notes,n_vocab):
    """ Create input and output sequences """
    
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    sequence_length = 50
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
       
    n_patterns = len(network_input)     
    
    # Reshape the input into a format compatible with LSTM layers and normalize
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    n_vocab = len(set(notes))
    network_input = network_input / n_vocab
    network_output = np_utils.to_categorical(network_output)
    
    return (network_input,network_output)

def create_network(network_input, n_vocab):
    
    """ Creates the structure of the neural network """
    
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    #model.load_weights('/Users/carlosestevezmartinez/Projects/Final-Project/weights/weights.35-4.1568.hdf5')
    
    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
     """ Generate notes from the neural network """
    
    # Pick a random sequence from the input to start the song
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate X notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction) # Numpy array of predictions
        result = int_to_note[index] # Choose note with the highest probability
        prediction_output.append(result) 

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


    def create_midi(prediction_output):
    """ Converts the output to notes and create a midi file """
    
    offset = 0
    output_notes = []

  
    for pattern in prediction_output:
        # Pattern is a chord
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

        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.SnareDrum()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not overlap
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    # Poner el nombre que quieras al archivo nuevo creado
    midi_stream.write('midi', fp='test_output3.mid')