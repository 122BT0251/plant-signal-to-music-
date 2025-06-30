import os
import io
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, request, send_file, jsonify
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from scipy.io.wavfile import read as wav_read
import ffmpeg

SOUNDFONT_URL = "https://github.com/urish/cynthion/raw/main/resources/FluidR3_GM.sf2"
SOUNDFONT_PATH = "FluidR3_GM.sf2"

app = Flask(__name__)

# MIDI Instrument Mapping (0–127)
instrument_map = {k: v for k, v in enumerate([
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1",
    "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar",
    "Guitar harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1",
    "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
    "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
    "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba",
    "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax",
    "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder",
    "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)",
    "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)",
    "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot"
])}
name_to_number = {v: k for k, v in instrument_map.items()}

# Load your plant signal data here
df = pd.read_excel("signal_data.xlsx")
signal = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values

def normalize_signal(signal, min_val=30, max_val=90):
    if np.max(signal) == np.min(signal):
        return np.full_like(signal, (min_val + max_val) // 2)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
    return (signal * (max_val - min_val) + min_val).astype(int)

def generate_midi(notes, filename, instrument_number, tempo_multiplier):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    bpm = 120 * tempo_multiplier
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))
    track.append(Message("program_change", program=instrument_number, time=0))

    note_length = 240
    gap = 0

    for i, note in enumerate(notes):
        time_on = gap if i != 0 else 0
        pitch = int(note)
        track.append(Message("note_on", note=pitch, velocity=64, time=time_on))
        track.append(Message("note_off", note=pitch, velocity=64, time=note_length))

    mid.save(filename)

def convert_wav_to_mp3(wav_path):
    buf = io.BytesIO()
    stream = ffmpeg.input(wav_path)
    stream = ffmpeg.output(stream, 'pipe:', format='mp3', acodec='libmp3lame')
    out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    buf.write(out)
    buf.seek(0)
    return buf

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    instrument = data.get("instrument", "Acoustic Grand Piano")
    tempo = float(data.get("tempo", 1.0))

    instrument_number = name_to_number.get(instrument, 0)
    notes = normalize_signal(signal)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mid_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:

        generate_midi(notes, mid_file.name, instrument_number, tempo)

        cmd = [
            'fluidsynth', '-ni', 'FluidR3_GM.sf2',
            mid_file.name, '-F', wav_file.name, '-r', '44100'
        ]
        os.system(' '.join(cmd))

        mp3_io = convert_wav_to_mp3(wav_file.name)

    return send_file(mp3_io, mimetype='audio/mpeg')

@app.route('/')
def home():
    return jsonify({"message": "Plant music backend is running!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
