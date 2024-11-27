import os
import string
import streamlit as st
import streamlit_mic_recorder

st.set_page_config(page_title="अनुवादक", layout="wide")

from streamlit_mic_recorder import mic_recorder
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
from configs import (
    CACHE_DIR,
    LANGUAGE,
    TASK,
    PRETRAINED_MODEL_NAME,
    KODES,
    DIGITS,
    MAX_AUDIO_DURATION,
)
from utils import (
    split_to_subwords,
    process,
    has_valid_duration,
    kataho_code_with_digits,
)


filepath = os.path.join("audio_files", "audio.wav")
audio_rec = None
audio_upload = None

st.title(":rainbow[Kataho Code Recognition] 🎤 :rainbow[अनुवादक]")


def model_pipeline():
    with st.spinner("Loading transcription model. It may take a while."):
        peft_config = PeftConfig.from_pretrained(
            PRETRAINED_MODEL_NAME, cache_dir=CACHE_DIR
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            # load_in_8bit=True,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        model = PeftModel.from_pretrained(model, PRETRAINED_MODEL_NAME)
        tokenizer = WhisperTokenizer.from_pretrained(
            str(peft_config.base_model_name_or_path),
            language=LANGUAGE,
            task=TASK,
            cache_dir=CACHE_DIR,
        )
        processor = WhisperProcessor.from_pretrained(
            str(peft_config.base_model_name_or_path),
            language=LANGUAGE,
            task=TASK,
            cache_dir=CACHE_DIR,
        )
        feature_extractor = processor.feature_extractor  # type: ignore
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)  # type: ignore
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor  # type: ignore
        )
        st.toast("Model Loaded")
        return pipe


def transcribe(filepath=filepath):
    if pipe := st.session_state.get("pipe"):
        result_np = pipe(filepath)
    else:
        st.toast("Pipeline is not loaded")
        st.stop()
    text: str = result_np["text"]  # type: ignore

    text = text.replace("�", "")

    for letter in string.ascii_letters:
        text = text.replace(letter, "")

    text_splitted = text.split()  # type: ignore
    _text_splitted = []
    for word in text_splitted:
        if word not in KODES + DIGITS:
            _text_splitted.extend(split_to_subwords(word, KODES + DIGITS))
        else:
            _text_splitted.append(word)
    text_splitted = _text_splitted

    if text_splitted:
        text = process(text_splitted)
    else:
        text = ""

    return text


if st.session_state.get("pipe") is None:
    st.session_state["pipe"] = model_pipeline()


os.makedirs("audio_files", exist_ok=True)

audio_mode = st.radio(
    "**Select audio input method**",
    ["Record", "Upload"],
    captions=["Record audio in browser", "Upload audio file from device"],
    horizontal=True,
)

if audio_mode == "Record":
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=False,
        format="wav",
        callback=None,
        args=(),
        kwargs={},
        key=None,
    )

    if audio is not None:
        audio = audio["bytes"]
        with open(filepath, "wb") as f:
            f.write(audio)
elif audio_mode == "Upload":
    audio = st.file_uploader("Upload audio file.", type=["wav"])

    if audio is not None:
        with open(filepath, "wb") as f:
            f.write(audio.getbuffer())


## check if audio if less than 30 seconds
if os.path.exists(filepath):
    st.audio(filepath)
    if not has_valid_duration(filepath):
        st.error(f"Audio duration is greater than {MAX_AUDIO_DURATION} seconds.")
        st.stop()

if audio and st.button("Transcribe"):
    with st.spinner("Transcribing..."):
        text = transcribe(filepath)
        text = kataho_code_with_digits(text)
    st.text("अनुवाद: " + text)

audio = None

if os.path.exists(filepath):
    os.remove(filepath)
