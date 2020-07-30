from music21 import *
import glob
import pickle
import numpy as np
from keras.utils import np_utils


def generatenotes():
    """ Extract songs from midi file and save all notes into string format """
    notes = []

    for file in glob.glob("midi_songs_train/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        
        if parts: 
            notes_to_parse = parts.parts[0].recurse()
        else: 
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
        
    return notes



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