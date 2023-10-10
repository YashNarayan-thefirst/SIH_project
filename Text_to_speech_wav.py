from gtts import gTTS

def tts(text):
    # Save the text-to-speech output as a WAV file
    tts_output = gTTS(text, lang='en', tld='us')
    tts_output.save('sample.wav')
txt = input("Enter text: ")
tts(txt)
