import torchaudio
import wave
import torch
import speechbrain as sb
from speechbrain.pretrained import DeepSpeech2
from speechbrain.pretrained import EncoderClassification
import pydub
def wavtopcm(file):#takes wav file as argument. converts into output.pcm. open that for data
    with wave.open(file) as file:
        #print('File opened!')
        sample_width = file.getsampwidth()
        frame_rate = file.getframerate()
        num_frames = file.getnframes()

        audio_data = file.readframes(num_frames)

    output_pcm_file = "output.pcm"
    with open(output_pcm_file, 'wb') as pcm_out:
        pcm_out.write(audio_data)


def transcribe_audio(audio_file, model_type="DS2"):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    print(waveform,sample_rate)
    # Initialize the model
    if model_type == "DS2":
        asr_model = DeepSpeech2.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="tmpdir_asr")
    #elif model_type == "EC":
        asr_model = EncoderClassification.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="tmpdir_asr")
    else:
        raise ValueError("Invalid model_type. Use 'DS2' or 'EC'.")

    # Perform speech recognition
    with torch.no_grad():  # Disable gradient computation during inference
        recognized_text = asr_model.decode_batch(waveform)

    return recognized_text

if __name__ == "__main__":
    audio_file = "sample.mp3"  # Replace with your audio file
    recognized_text = transcribe_audio(audio_file)
    print("Recognized Text:")
    print(recognized_text)
