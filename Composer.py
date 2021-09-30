import os
import pickle
import time
from re import X
from urllib.request import urlopen, urlretrieve

import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as np_utils
from bs4 import BeautifulSoup
from music21 import (chord, converter, instrument, interval, key, note, pitch,
                     stream)
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Input,
                                     concatenate)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import (not_equal,
                                                not_equal_eager_fallback,
                                                xdivy_eager_fallback, xlog1py)


# Functions to save an load lists from pickle files
def save_list(list_to_save, folder_name, file_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print('Making Directory:' + str(folder_name))
    with open(folder_name + '/' + file_name, 'wb') as fp:
        pickle.dump(list_to_save, fp)

def load_list(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'rb') as fp:
        return pickle.load(fp)


def Scrape(dir):
## This Function scrapes Mutopia website for Jazz Piano Midi Files.
## Change the URL to get different instruments and styles.

    # Directory to save the Midi files
    save_dir = dir

    # Url is plsit into two so we can go through the pages of the search results
    url0 = 'https://www.mutopiaproject.org/cgibin/make-table.cgi?startat='
    url1 = '&searchingfor=&Composer=&Instrument=Piano&Style=Jazz&collection=&id=&solo=1&recent=&timelength=1&timeunit=week&lilyversion=&preview='

    # Init values
    song_number = 0
    link_count = 10

    file_name = 0

    # main loop
    while link_count > 0:
        #finds the correct page of search results
        url = url0 + str(song_number) + url1
        html = urlopen(url)
        soup = BeautifulSoup(html.read())
        # Finds all the links on the page
        links = soup.find_all('a')
        link_count = 0

        for link in links:
            href = link['href']
            # Find all links with a .mid in them
            if href.find('.mid') >= 0:
                link_count = link_count + 1
                #Download that link
                urlretrieve(href, dir+str(file_name)+'.mid' )
                file_name += 1

        #+10 since there are 10 results on each page
        song_number += 10
        # Small wait to be nice to the website
        time.sleep(10.0)



def ProcessMidis(dir, seq_len):
    ## function that takes all the midi files in a directory and converts them to NN input

    save_dir = dir

    song_list = os.listdir(save_dir)

    original_scores = []

    # Adds the parsed songs to a list
    for song in song_list:
        print(song)
        midi = converter.parse(save_dir+song)
        key = midi.analyze('key')
        if str(key) == 'C major' or str(key) == 'c minor':
            original_scores.append(midi)

    # Remove polyphonic music (multiple instruments)
    # This function checks for monophonic music
    # If we don't do this, notes from multiple instruments will be combined into chords
    def monophonic(stream):
        try:
            length = len(instrument.partitionByInstrument(stream).parts)
        except:
            length = 0
            print(str(stream) + 'is NOT monotonic')
        return length == 1

    # Loops through songs, checks they are monophonic then chordifies and adds to list
    original_scores_chordified = []
    for song in original_scores:
        if monophonic(song):
            print(str(song)+'is monotonic')
            song.chordify()
            # flat the notes, not entirely sure what this does but it's necessary
            original_scores_chordified.append(song.flat.notes)

    original_scores = original_scores_chordified

    #print(original_scores)

    # Now we need to extract the notes, chords and durations from the songs
    original_chords = [[] for _ in original_scores] #empty list of lists
    original_durations = [[] for _ in original_scores]
    original_keys = []

    for i, song in enumerate(original_scores):
        # Save the key of the song
        original_keys.append(str(song.analyze('key')))
        # Loop through the notes and chords
        for element in song:
            # if note
            if isinstance(element, note.Note):
                # add note
                original_chords[i].append(str(element.pitch))
                original_durations[i].append(element.duration.quarterLength)
            # if chord
            elif isinstance(element, chord.Chord):
                # add all notes making up chord
                original_chords[i].append('.'.join(str(n) for n in element.pitches))
                original_durations[i].append(element.duration.quarterLength)

        print(str(i))


    # I am going to keep all the key signatures for now unlike tutorial

    # Identify unique notes and chords and create dictionaries to convert to ints
    unique_chords = np.unique([i for s in original_chords for i in s])
    chord_to_int = dict(zip(unique_chords, list(range(0, len(unique_chords)))))

    # Map durations to ints too
    unique_dur = np.unique([i for s in original_durations for i in s])
    dur_to_int = dict(zip(unique_dur, list(range(0, len(unique_dur)))))

    # We also need dictionaries to convert the other way
    int_to_chord = {i: c for c, i in chord_to_int.items()}
    int_to_dur = {i: c for c, i in dur_to_int.items()}

    # Lastly we can make our training sequences and target notes

    train_chords = []
    train_dur = []

    target_chords = []
    target_dur = []

    # loop through the chords
    for s in range(len(original_chords)):
        # create a list of ints from the chord list
        chord_list = [chord_to_int[c] for c in original_chords[s]]
        # make sequences 32 in length and add to the training lists
        for i in range(len(chord_list) - seq_len):
            train_chords.append(chord_list[i:i+seq_len])
            target_chords.append(chord_list[i+1])
            
    for d in range(len(original_durations)):
        dur_list = [dur_to_int[s] for s in original_durations[d]]
        for i in range(len(dur_list)-seq_len):
            train_dur.append(dur_list[i:i+seq_len])
            target_dur.append(dur_list[i+1])

    # Reshape to fit LSTM
    input_chords = np.reshape(np.array(train_chords), (len(train_chords), seq_len,1))
    input_dur = np.reshape(np.array(train_dur), (len(train_dur), seq_len, 1))
    # Normalise these
    input_chords = input_chords / float(len(unique_chords))
    input_dur = input_dur / float(len(unique_dur))
    # Make target notes categorical
    
    target_dur = np_utils.to_categorical(target_dur)
    target_chords = np_utils.to_categorical(target_chords)

    no_chords = target_chords.shape[1]
    no_dur = target_dur.shape[1]

    # Save all the important data structures so this whole func doesnt have to run every time
    print('Saving Data...')
    save_list(input_chords, 'Processed_Data', 'input_chords')
    save_list(input_dur, 'Processed_Data', 'input_dur')
    save_list(target_chords, 'Processed_Data', 'target_chords')
    save_list(target_dur, 'Processed_Data', 'target_dur')
    save_list(int_to_chord, 'Processed_Data', 'int_to_chord')
    save_list(int_to_dur, 'Processed_Data', 'int_to_dur')
    print('Saving Complete...')

    return (input_chords, input_dur, target_chords, target_dur, no_chords, no_dur, int_to_chord, int_to_dur)


def create_lstm(input_chords, no_chords, input_dur, no_dur):
    
    # Notes Input branch
    chord_input_layer = Input(shape=(input_chords.shape[1], input_chords.shape[2]))
    chord_input = LSTM(
        256,
        input_shape = (input_chords.shape[1], input_chords.shape[2]),
        return_sequences = True
    )(chord_input_layer)
    # Dropout randomly sets inputs to 0 to help prevent overfitting
    chord_input = Dropout(0.2)(chord_input)

    # Duration input branch
    dur_input_layer = Input(shape=(input_dur.shape[1], input_dur.shape[2]))
    dur_input = LSTM(
        256,
        input_shape = (input_dur.shape[1], input_dur.shape[2]),
        return_sequences = True
    )(dur_input_layer)
    # Dropout to prevent overfitting
    dur_input = Dropout(0.2)(dur_input)

    # Concatenate the inputs
    inputs = concatenate([chord_input, dur_input])

    # Input goes through another LSTM
    x = LSTM(512, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(512)(x)
    # batch norm makes the output closer to normal dist.
    x = BatchNorm()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)

    # Split into 2 branches again
    output_chords = Dense(128, activation='relu')(x)
    output_chords = BatchNorm()(output_chords)
    output_chords = Dropout(0.3)(output_chords)
    output_chords = Dense(no_chords, activation='softmax', name='Note')(output_chords)

    output_dur = Dense(128, activation='relu')(x)
    output_dur = BatchNorm()(output_dur)
    output_dur = Dropout(0.3)(output_dur)
    output_dur = Dense(no_dur, activation='softmax', name='Duration')(output_dur)

    # Define model with inputs and outputs
    model = Model(inputs= [chord_input_layer, dur_input_layer], outputs = [output_chords, output_dur])

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def make_or_restore(checkpoint_dir, input_chords, no_chords, input_dur, no_dur):
    ## Function that will restore the model to the most recent checkpoint or make a fresh one

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:

        epoch_list = []
        # Finds the epoch number from the string of its name
        for c in checkpoints:
            epoch_list.append(find_initial_epoch(str(c)))

        #finds the index of the largest epoch number and loads it
        index = epoch_list.index(max(epoch_list))
        latest_checkpoint = checkpoints[index]
        print("Restoring from", latest_checkpoint)

        initial_epoch = max(epoch_list)

        return load_model(latest_checkpoint), initial_epoch
    
    print("Creating a new model")
    initial_epoch = 0
    return create_lstm(input_chords, no_chords, input_dur, no_dur), initial_epoch


def find_initial_epoch(checkpoint_str):
    ## Function that finds the initial epoch from the saved model name
    
    # This loop finds the second '-' in the str
    for i in range(2):
        refs = checkpoint_str.find('-')
        # the start of the epoch number is the next char after
        checkpoint_str = checkpoint_str[refs+1:]

    # then find the red for the following '-' after the epoch number
    refs = checkpoint_str.find('-')
    # strip away the rest of the string
    checkpoint_str = checkpoint_str[:refs]
    checkpoint_str = checkpoint_str.lstrip('0')

    initial_epoch = int(checkpoint_str)
    
    return initial_epoch


def train(model, input_chords, input_dur, target_chords, target_dur, initial_epoch):

    checkpoint_dir = "./weights"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min',
	)
    history = History()
    callbacks_list = [checkpoint, history]

    model.fit([input_chords, input_dur],[target_chords, target_dur], 
            epochs=1000, batch_size=64, callbacks=callbacks_list, 
            verbose=1, initial_epoch = initial_epoch)


def generate_seq(model, chord_pattern, dur_pattern, int_to_note, int_to_dur, temp):

    prediction_output = []

    for note_index in range(400):

        chord_prediction_input = np.reshape(chord_pattern, (1, len(chord_pattern), 1))
        predictedNote = chord_prediction_input[-1][-1][-1]

        dur_prediction_input = np.reshape(dur_pattern, (1, len(dur_pattern), 1))
    
        prediction = model.predict([chord_prediction_input, dur_prediction_input])

        # Use the sample function to add some extra randomness
        prediction[0] = sample(prediction[0], temp)
        prediction[1] = sample(prediction[1], temp)

        chord_index = np.argmax(prediction[0])
        chord_result = int_to_note[chord_index]

        dur_index = np.argmax(prediction[1])
        dur_result = int_to_dur[dur_index]

        print("Next note: " + str(chord_result) + " - Duration: " + str(dur_result))
		
        prediction_output.append([chord_result, dur_result])

        chord_pattern = np.append(chord_pattern, chord_index)
        dur_pattern = np.append(dur_pattern, dur_index)

        chord_pattern = chord_pattern[1:len(chord_pattern)]
        dur_pattern = dur_pattern[1:len(dur_pattern)]
    
    return prediction_output


def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    
    # Adds a lil bit of randomness
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds[0,:], 1)

    return probas


def generate_midi(predicted_notes, midi_name):

    offset = 0
    output_notes = []

    for pred_note in predicted_notes:
        current_chord = pred_note[0]
        current_dur = pred_note[1]

        if('.' in current_chord) or current_chord.isdigit():
            notes_in_chord = current_chord.split('.')
            notes = []

            for n in notes_in_chord:
                new_note = note.Note(n)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                #print('chord:'+str(notes))
            
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = current_dur
            new_chord.volume.velocityScalar =0.5
            output_notes.append(new_chord)
        
        else:
            new_note = note.Note(current_chord)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = current_dur
            new_note.volume.velocityScalar =0.5
            output_notes.append(new_note)

        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=midi_name)


def prep_data():

    print('Loading Data...')
    input_chords = load_list('Processed_Data', 'input_chords')
    input_dur = load_list('Processed_Data', 'input_dur')
    target_chords = load_list('Processed_Data', 'target_chords')
    target_dur = load_list('Processed_Data', 'target_dur')
    int_to_chord = load_list('Processed_Data', 'int_to_chord')
    int_to_dur = load_list('Processed_Data', 'int_to_dur')
    print('... Loading Complete')

    print(target_chords.shape)
    print(target_dur.shape)

    no_chords = target_chords.shape[1]
    no_dur = target_dur.shape[1]

    print('Number of chords:'+ str(no_chords))
    print('Number of durations:'+ str(no_dur))

    return input_chords, input_dur, target_chords, target_dur, int_to_chord, int_to_dur, no_chords, no_dur

def main():

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    #Scrape('piano/')
    #ProcessMidis('piano/', 100)

    # Load the data, comment out ProcessMidis once the Processed_Data folder is full
    input_chords, input_dur, target_chords, target_dur, int_to_chord, int_to_dur, no_chords, no_dur = prep_data()
    

    model, initial_epoch = make_or_restore('./weights', input_chords, no_chords, input_dur, no_dur)
    #train(model, input_chords, input_dur, target_chords, target_dur, initial_epoch)

    # Choose random start sequence
    start_chords = np.random.randint(0, len(input_chords)-1)
    start_dur = np.random.randint(0, len(input_dur) -1)

    chord_pattern = input_chords[start_chords]
    dur_pattern = input_dur[start_dur]

    predicted_notes = generate_seq(model, chord_pattern, dur_pattern, int_to_chord, int_to_dur, 0.4)

    generate_midi(predicted_notes, 'test.mid')

    # Uncomment this to save a diagram of the model
    #img_file = 'model.png'
    #np_utils.plot_model(model, to_file=img_file, show_shapes=True)


if __name__ == '__main__':
	main()
