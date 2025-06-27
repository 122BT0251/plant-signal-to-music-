
from flask import Flask, request, send_file, jsonify
import pandas as pd
import numpy as np
import tempfile
import subprocess
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from pydub import AudioSegment
from pydub.generators import Sine
import io
import os
import urllib.request

SOUNDFONT_URL = "https://github.com/urish/cynthion/raw/main/resources/FluidR3_GM.sf2"
SOUNDFONT_PATH = "FluidR3_GM.sf2"

# Download the file if it's not already there
if not os.path.exists(SOUNDFONT_PATH):
    print("Downloading FluidR3_GM.sf2...")
    urllib.request.urlretrieve(SOUNDFONT_URL, SOUNDFONT_PATH)

app = Flask(__name__)

instrument_map = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano", 3: "Honky-tonk Piano",
    4: "Electric Piano 1", 5: "Electric Piano 2", 6: "Harpsichord", 7: "Clavinet",
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone",
    12: "Marimba", 13: "Xylophone", 14: "Tubular Bells", 15: "Dulcimer",
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ", 19: "Church Organ",
    20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)", 27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)", 29: "Overdriven Guitar", 30: "Distortion Guitar", 31: "Guitar harmonics",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)", 35: "Fretless Bass",
    36: "Slap Bass 1", 37: "Slap Bass 2", 38: "Synth Bass 1", 39: "Synth Bass 2",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    44: "Tremolo Strings", 45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani",
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)", 83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)", 85: "Lead 6 (voice)", 86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)", 101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto",
    108: "Kalimba", 109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock",
    116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum", 119: "Reverse Cymbal",
    120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore", 123: "Bird Tweet",
    124: "Telephone Ring", 125: "Helicopter", 126: "Applause", 127: "Gunshot"
}
name_to_number = {v: k for k, v in instrument_map.items()}

def normalize_signal(signal, min_val=30, max_val=90):
    signal = np.array(signal)
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
    gap_between_notes = 0

    for i, note in enumerate(notes):
        time_on = gap_between_notes if i != 0 else 0
        track.append(Message("note_on", note=int(note), velocity=64, time=time_on))
        track.append(Message("note_off", note=int(note), velocity=64, time=note_length))

    mid.save(filename)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    instrument = data.get("instrument", "Acoustic Grand Piano")
    tempo = float(data.get("tempo", 1.0))
    gain_db = int(data.get("gain", 10))
    reverb = data.get("reverb", True)
    spatial = data.get("spatial", "Center")
    timbre = data.get("timbre", "None")
    raw_signal = data.get("signal", [])

    signal = normalize_signal(raw_signal)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:

        generate_midi(signal, midi_file.name, name_to_number[instrument], tempo)

        reverb_flags = [] if reverb else ['-R', '0']
        subprocess.run(
            ['fluidsynth', '-ni'] + reverb_flags + [
                '/app/FluidR3_GM.sf2',
                midi_file.name,
                '-F', wav_file.name,
                '-r', '44100'
            ], check=True
        )

        audio = AudioSegment.from_wav(wav_file.name)

        duration_ms = len(audio)
        drone = Sine(220).to_audio_segment(duration=duration_ms).apply_gain(-30).low_pass_filter(400)
        mixed_audio = audio.overlay(drone)
        mixed_audio += gain_db

        if spatial == "Left":
            mixed_audio = mixed_audio.pan(-1.0)
        elif spatial == "Right":
            mixed_audio = mixed_audio.pan(1.0)

        if timbre == "Warm":
            mixed_audio = mixed_audio.low_pass_filter(2000)
        elif timbre == "Smooth":
            mixed_audio = mixed_audio.high_pass_filter(500).low_pass_filter(5000)
        elif timbre == "Rich":
            mixed_audio = mixed_audio + mixed_audio.high_pass_filter(1000)

        buf = io.BytesIO()
        mixed_audio.export(buf, format="mp3")
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg")

@app.route('/')
def home():
    return jsonify({"message": "Backend is running"}), 200

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
