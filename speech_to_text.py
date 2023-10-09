import torchaudio
import torch
from speechbrain.pretrained import DeepSpeech2
from speechbrain.pretrained import EncoderClassification

def transcribe_audio(audio_file, model_type="DS2"):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)

    # Initialize the model
    if model_type == "DS2":
        asr_model = DeepSpeech2.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="tmpdir_asr")
    elif model_type == "EC":
        asr_model = EncoderClassification.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="tmpdir_asr")
    else:
        raise ValueError("Invalid model_type. Use 'DS2' or 'EC'.")

    # Perform speech recognition
    with torch.no_grad():  # Disable gradient computation during inference
        recognized_text = asr_model.decode_batch(waveform)

    return recognized_text

if __name__ == "__main__":
    audio_file = "sample.wav"  # Replace with your audio file
    recognized_text = transcribe_audio(audio_file)
    print("Recognized Text:")
    print(recognized_text)
