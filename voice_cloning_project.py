from TTS.api import TTS
import os



# Load the multi-speaker, multi-language model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True)

# Inputs
voice_sample = "voice_sample.wav"
output_path = "anza_clone.wav"
text = "Hello, this is my cloned voice speaking using AI!"
language = "en"  

# Generate audio
tts.tts_to_file(
    text=text,
    speaker_wav=voice_sample,
    language=language,  
    file_path=output_path
)
# Check if the output file was created
print(f"Voice cloned! Output saved at {output_path}")
