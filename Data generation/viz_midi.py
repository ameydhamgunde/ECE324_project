import pretty_midi
import os
os.chdir(r"C:\Users\roryg\Documents\Engsci\2023\ECE324\out\out")



mid_data = pretty_midi.PrettyMIDI("btvn-rnd.mid")

piano = mid_data.instruments[0]
guitar = mid_data.instruments[1]

instr = [piano, guitar]
instruments = ["piano_pred", "guitar_pred"]
for instrument, instrument_name in zip(instr, instruments):
    mid = pretty_midi.PrettyMIDI()
    prog = instrument.program
    new_instrument = pretty_midi.Instrument(program = prog)
    for note in instrument.notes:
        note_inst = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end)
        new_instrument.notes.append(note_inst)
    mid.instruments.append(new_instrument)
    mid.write(os.path.join(os.getcwd(), f"{instrument_name}.mid"))



# add the tracks together
os.chdir(r"C:\Users\roryg\Documents\Engsci\2023\ECE324\out\out\srop18_pred")
for track in ["guitar", "piano"]:
    pred_mid = pretty_midi.PrettyMIDI(track+"_pred.mid")
    mid = pretty_midi.PrettyMIDI(track+".mid")
    new_instr = pretty_midi.Instrument(program = 50) #this one is on top
    for note in pred_mid.instruments[0].notes: #for notes in the predicted track
        note_inst = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end)
        new_instr.notes.append(note_inst)
    mid.instruments.append(new_instr)
    mid.write(os.path.join(os.getcwd(), f"{track}_missing.mid"))

for track in ["guitar", "piano"]:
    pred_mid = pretty_midi.PrettyMIDI(track+"_pred.mid")
    mid = pretty_midi.PrettyMIDI(track+".mid")
    new_instr = pretty_midi.Instrument(program = 50) #this one is on top
    for note in mid.instruments[0].notes: #for notes in the predicted track
        note_inst = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end)
        new_instr.notes.append(note_inst)
    pred_mid.instruments.append(new_instr)
    pred_mid.write(os.path.join(os.getcwd(), f"{track}_extra.mid"))
