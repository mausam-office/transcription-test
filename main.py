import os
import streamlit as st
from datetime import datetime
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
from peft import PeftModel, PeftConfig

from configs import (
    CACHE_DIR, LANGUAGE, TASK, PRETRAINED_MODEL_NAME, KODES, DIGITS, 
    MAX_AUDIO_DURATION
)
from utils import split_to_subwords, process, has_valid_duration


st.title(":blue[Nepali Speech Recognition] :sunglasses:")

def model_pipeline():
    peft_config = PeftConfig.from_pretrained(PRETRAINED_MODEL_NAME, cache_dir=CACHE_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, 
        # load_in_8bit=True, 
        device_map="auto", 
        cache_dir=CACHE_DIR
    )
    model = PeftModel.from_pretrained(model, PRETRAINED_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        str(peft_config.base_model_name_or_path), language=LANGUAGE, task=TASK, cache_dir=CACHE_DIR 
    )
    processor = WhisperProcessor.from_pretrained(
        str(peft_config.base_model_name_or_path), language=LANGUAGE, task=TASK, cache_dir=CACHE_DIR
    )
    feature_extractor = processor.feature_extractor # type: ignore
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK) # type: ignore
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model, tokenizer=tokenizer, feature_extractor=feature_extractor # type: ignore
    )
    return pipe


if st.session_state.get('pipe') is None:
    st.session_state['pipe'] = model_pipeline()

uploaded_audio = st.file_uploader("Upload audio file.", type=['wav'])

if uploaded_audio is not None:
    file_name, file_extension = os.path.splitext(uploaded_audio.name)
    file_name += f"-{str(datetime.now()).replace(':','-')}"
    
    filepath = os.path.join("audio_files", file_name + file_extension)
    os.makedirs("audio_files", exist_ok=True)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    
    ## check if audio if less than 30 seconds  
    if not has_valid_duration(filepath):
        st.error(f"Audio duration is greater than {MAX_AUDIO_DURATION} seconds.")
        st.stop()
    
    st.audio(filepath)
    
    if pipe:=st.session_state.get('pipe'):
        result_np = pipe(filepath)
    else:
        st.toast("Pipeline is not loaded")
        st.stop()
    text: str = result_np['text'] # type: ignore
    
    text = text.replace('�', '') 
    
    words = []
    
    text_splitted = text.split() # type: ignore
    _text_spiltted = []
    for word in text_splitted:
        _text_spiltted.extend(split_to_subwords(word, KODES + DIGITS))
    
    text_splitted = text_splitted
    
    if text_splitted:
        text = process(text_splitted)
    else:
        text = ''
    
    st.text("Transcription: " + text)
    

