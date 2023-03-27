import sf2_loader as sf
import os
from scipy import signal
from scipy import io
import numpy as np
ECE324_dir = r"C:\\Users\\ameyd\\Documents\\Assignments\\ECE324_project\\" #CHANGE THIS


os.chdir(ECE324_dir)
clarinet = sf.sf2_loader(r".\Soundfonts\Clarinet-20190818.sf2")
guitar = sf.sf2_loader(r".\\Soundfonts\\SpanishClassicalGuitar-20190618.sf2")
piano = sf.sf2_loader(r".\\Soundfonts\\UprightPianoKW-20220221.sf2")

instruments = {"clarinet":clarinet, "guitar":guitar,"piano":piano}
num_instruments = 3

pieces = os.listdir(r".\MIDI") #read midi classical pieces
num_pieces = len(pieces)

concert_response = np.load("concert.npy") #impulse response of concert hall
noised_imgs = np.random.choice(indices, size = 100 replace = False)
white_noise = np.random.normal(0,1,100) #white noise of length 100. Duplicate if longer needed
amplitude_imgs = np.random.choice(indices, size = 50, replace = False)
amplitude_adj = np.random.random(50) * max_amplitude

data_types = ["all instr mixed pieces","all instr same piece"]
folder_names = {"all instr mixed pieces": "mix", "all instr same piece": "same"}

if "data" not in os.listdir(): #make folder called data
    os.mkdir("./data")

def combine_tracks(directory, num_instruments):
    os.chdir(directory)
    tracks = []
    max_len = 0
    for instrument in os.listdir():
        _, music_signal = io.wavfile.read(instrument)
        tracks.append(music_signal)
        max_len = max(max_len, music_signal.shape[0])
    combined_track = np.zeros((max_len, num_instruments, 1))
    for id, instrument_track in zip(range(num_instruments), tracks):
        combined_track[:,id, 0] = np.pad(instrument_track[:,0], pad_width = (0, max_len - instrument_track.shape[0]),mode = "constant")
    os.chdir(ECE324_dir)
    return combined_track.sum(axis = 1).astype(np.dtype("i2"))





for music_data_type in data_types:
    folder_name = folder_names[music_data_type]

    if folder_name == "mix":
        if "mix" not in os.listdir("./data"):
            os.mkdir("./data/mix/")
        for iter in range(num_pieces):
            if str(iter) not in os.listdir("./data/mix"):
                os.mkdir("./data/mix/"+str(iter))
                os.mkdir("./data/mix/"+str(iter)+"/"+"stems") #folder holds separated tracks

            for instrument in instruments:
                print(instrument)
                piece = np.random.choice(pieces)

                instruments[instrument].export_midi_file(r"./MIDI/"+piece, name = r"./data/mix/"+str(iter)+"/"+"stems"+"/"+instrument+".wav", format = "wav")
             #Now combine all three into a combined track

            combined_track = combine_tracks(r"./data/mix/"+str(iter)+"/"+"stems",num_instruments)
            io.wavfile.write(r"./data/mix/"+str(iter)+"/all.wav",44100,combined_track)


    if folder_name == "same":
        if "same" not in os.listdir("./data"):
            os.mkdir("./data/same/")
        for iter in range(num_pieces):
            if str(iter) not in os.listdir("./data/same"):
                os.mkdir("./data/same/"+str(iter))
                os.mkdir("./data/same/"+str(iter)+"/"+"stems") #folder holds separated tracks

            for instrument in instruments:
                piece = pieces[iter]
                instruments[instrument].export_midi_file("./MIDI/"+piece, name = r"./data/same/"+str(iter)+"/"+"stems"+"/"+instrument+".wav", format = "wav")


            #Now combine all three into a combined track
            combined_track = combine_tracks(r"./data/same/"+str(iter)+"/"+"stems",num_instruments)

            io.wavfile.write(r"./data/same/"+str(iter)+"/all.wav",44100,combined_track)













