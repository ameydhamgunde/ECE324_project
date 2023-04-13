import sf2_loader as sf
import os
from scipy import signal
from scipy import io
import numpy as np

ECE324_dir = r"C:\Users\roryg\Documents\Engsci\2023\ECE324" #CHANGE THIS


os.chdir(ECE324_dir)
guitar = sf.sf2_loader(r".\Soundfonts\SpanishClassicalGuitar-20190618.sf2")
piano = sf.sf2_loader(r".\Soundfonts\UprightPianoKW-20220221.sf2")


instruments = {"guitar":guitar, "piano":piano}
num_instruments = 2

def combine_tracks(directory, num_instruments):
    os.chdir(directory)
    tracks = []
    max_len = 0
    for instrument in os.listdir():
        if instrument != "MIDI":
            _, music_signal = io.wavfile.read(instrument)
            tracks.append(music_signal)
            max_len = max(max_len, music_signal.shape[0])
    combined_track = np.zeros((max_len, num_instruments, 1))
    for id, instrument_track in zip(range(num_instruments), tracks):
        combined_track[:,id, 0] = np.pad(instrument_track[:,0], pad_width = (0, max_len - instrument_track.shape[0]),mode = "constant")
    os.chdir(ECE324_dir)
    return combined_track.sum(axis = 1).astype(np.dtype("i2"))



for piece in os.listdir("MIDI"):
    for instrument in instruments.keys():
        instruments[instrument].export_midi_file(fr"./MIDI/{piece}/MIDI/{instrument}.mid", name = fr"./MIDI/{piece}/{instrument}.wav", format = "wav")
    combined_track = combine_tracks(fr"./MIDI/{piece}/",num_instruments)
    io.wavfile.write(fr"./MIDI/{piece}/mix.wav",44100,combined_track)
