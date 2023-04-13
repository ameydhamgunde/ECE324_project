import pretty_midi
import os
os.chdir(r"C:\Users\roryg\Documents\Engsci\2023\ECE324\MIDI")
# Create a PrettyMIDI object
songs = ["1812-rev.mid", "btvn-rnd.mid", "chet1009.mid","chet2512.mid", "furelise.mid", "mozeine.mid", "mozk246b.mid", "mozk309c.mid", "nut5chin.mid", "srop18.mid"]
for song in songs:
    folder_name = song.split(".")[0]
    if folder_name not in os.listdir():
        os.mkdir(folder_name)

    piano_mid = pretty_midi.PrettyMIDI()
    violin_mid = pretty_midi.PrettyMIDI()
    #Create a piano instrument instance
    piano_prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program = piano_prog)
    #Create a violin instrument instance
    violin_prog = pretty_midi.instrument_name_to_program("Acoustic Guitar (nylon)") #violin is not acoustic guitar!
    violin = pretty_midi.Instrument(program = violin_prog)

    mid_data = pretty_midi.PrettyMIDI(song)

    i = 0
    for instr in mid_data.instruments:
        if instr.is_drum == False:
            for note in instr.notes:
                piano_note_inst = pretty_midi.Note(
            velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end)
                violin_note_inst = pretty_midi.Note(velocity = note.velocity, pitch = note.pitch - 24 if note.pitch > 24 else note.pitch, start = note.start, end = note.end)
                piano.notes.append(piano_note_inst)
                violin.notes.append(violin_note_inst)
    piano_mid.instruments.append(piano)
    violin_mid.instruments.append(violin)
    if "MIDI" not in os.listdir(folder_name):
        os.mkdir(f"./{folder_name}/MIDI")
    piano_mid.write(os.path.join(folder_name,"MIDI", "piano.mid"))
    violin_mid.write(os.path.join(folder_name,"MIDI","guitar.mid"))
