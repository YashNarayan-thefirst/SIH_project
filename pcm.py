import wave
from pydub import AudioSegment
def main():
    mp3_file = "sample.mp3"
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = "sample.wav"
    audio.export(wav_file, format="wav")
    print(f"MP3 file '{mp3_file}' has been converted to WAV file '{wav_file}'.")
    file_path = "sample.wav"
    with wave.open(file_path) as file:
        #print('File opened!')
        sample_width = file.getsampwidth()
        frame_rate = file.getframerate()
        num_frames = file.getnframes()
        audio_data = file.readframes(num_frames)
    output_pcm_file = "output.pcm"
    with open(output_pcm_file, 'wb') as pcm_out:
        pcm_out.write(audio_data)

if __name__ == "__main__":
    main()