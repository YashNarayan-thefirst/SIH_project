import torch
import torchaudio
from pydub import AudioSegment
from pydub.playback import play
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Convert MP3 to WAV
def convert_mp3_to_wav(input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format="wav")

def transcribe_audio(audio_file, model_type="facebook/wav2vec2-base-960h"):
    # Convert MP3 to WAV
    wav_audio_file = "sample.wav"
    convert_mp3_to_wav(audio_file, wav_audio_file)

    # Load the pre-trained model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_type)
    model = Wav2Vec2ForCTC.from_pretrained(model_type)

    # Load the audio file and perform inference
    waveform, sample_rate = torchaudio.load(wav_audio_file)

    # Tokenize and convert waveform to input format
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values

    # Perform speech recognition
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription

if __name__ == "__main__":
    audio_file = "sample.mp3"  # Replace with your MP3 audio file
    recognized_text = transcribe_audio(audio_file)
    print("Recognized Text:")
    print(recognized_text)
