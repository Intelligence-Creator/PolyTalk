import streamlit as st
import whisper
from translatepy import Translator
from gtts import gTTS
import sounddevice as sd
import scipy.io.wavfile as wav
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Top 20 spoken languages with their ISO codes
top_languages = {
    "en": "English", "zh": "Chinese", "hi": "Hindi", "es": "Spanish",
    "fr": "French", "ar": "Arabic", "bn": "Bengali", "ru": "Russian",
    "pt": "Portuguese", "ur": "Urdu", "id": "Indonesian", "de": "German",
    "ja": "Japanese", "sw": "Swahili", "mr": "Marathi", "te": "Telugu",
    "tr": "Turkish", "vi": "Vietnamese", "ko": "Korean", "it": "Italian"
}

st.title("Multilingual Voice Translator")
st.write("Record your voice and translate it into up to 4 languages")

# Language selector
selected_langs = st.multiselect("Select up to 4 target languages:",
                                 options=list(top_languages.keys()),
                                 format_func=lambda x: top_languages[x],
                                 max_selections=4,
                                 default=["es", "fr", "de", "ar"])

# Record button
if st.button("Record and Translate"):
    duration = 5  # seconds
    fs = 44100
    st.write("Recording for 5 seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav.write(tmp_file.name, fs, recording)
        audio_path = tmp_file.name

    st.write("Transcribing...")
    result = model.transcribe(audio_path)
    original_text = result["text"]
    source_lang = result["language"]
    st.write(f"Detected [{top_languages.get(source_lang, source_lang)}]: {original_text}")

    translator = Translator()

    for lang_code in selected_langs:
        translated = translator.translate(original_text, lang_code).result
        st.write(f"**{top_languages[lang_code]}**: {translated}")
        try:
            tts = gTTS(text=translated, lang=lang_code)
            tts_path = tempfile.mktemp(suffix=".mp3")
            tts.save(tts_path)
            st.audio(tts_path, format="audio/mp3")
        except Exception as e:
            st.error(f"Text-to-speech failed for {top_languages[lang_code]}: {e}")

st.markdown('---')
st.markdown('**PolyTalk** is free & open to all. [Donate here](https://ko-fi.com/the_intelligence_creator).  
Built by The Intelligence Creator.')