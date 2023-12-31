import os
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import speech_recognition as sr
import tensorflow as tf
import gpt_2_simple as gpt2
import zipfile  
import sttv2
global sess
sess = gpt2.start_tf_sess()
fonts = lambda a:("Inter ExtraBold", a * -1,'bold')
o=lambda x:x/1920
p=lambda x:x/1080
fr = lambda x: rf"{os.getcwd()}\assets\{x}"
t=lambda s:8*s//10

extraction_dir = f'{os.getcwd()}'  # Change to your desired directory
def extract_zip(zip_file, target_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

for i in ['checkpoint.zip','gpt2_models.zip','assets.zip']:
    extract_zip(i,extraction_dir)

gpt2.load_gpt2(sess, model_name='124M',model_dir='gpt2_models')

def resp(inp):
    res = gpt2.generate(sess,'124M',prefix = inp, return_as_list= True)[0]
    #print(res)
    return res

def tts(text):
    tts_output = gTTS(text, lang='en', tld='us')
    tts_output.save('sample.mp3')
    audio = AudioSegment.from_mp3('sample.mp3')
    audio.export('sample.wav', format="wav")

def stt(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
    # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        return(text)

def dest():
    for i in ['sb','msgbox','button3']:
        try:
            exec(f'{i}.destroy()')
        except:pass


def enter(event):
    if event.keysym == 'Return' and not event.state & 0x0001:# Check if the Shift key is also pressed (Shift+Enter)
        submit()
        return "break"
    else:
        entry_1.event_generate('<<Return>>')

def play_audio(audio_file):
    audio = AudioSegment.from_file(audio_file)
    play(audio)


def submit(inp='',file_path=''):
    dest()
    inp = (entry_1.get(1.0, "end-1c")) if inp != ''  else inp#user input
    output_display= resp(inp) #output recieved from chatbot
    tts(output_display)
    msgbox = Text(window,wrap='word',font=fonts(24))
    msgbox.insert(END,output_display)
    msgbox.place(relx=o(1165),rely=p(330),height=(250),width=(450))
    sb = Scrollbar(window,orient= "vertical")
    sb.config(command=msgbox.yview)
    sb.place(relx=o(1730),rely=p(330),height = (250))
    button_image_3 = ImageTk.PhotoImage((Image.open(fr("button_3.png")).resize((t(135),t(91)))))
    button3 = Button(image=button_image_3,borderwidth=0,highlightthickness=0,command=lambda: play_audio('sample.wav') if file_path=='' else play_audio(file_path),relief="flat")
    button3.image = button_image_3
    button3.place(relx=o(1780.0),rely=p(560.0),width=t(135.0),height=t(91.0))
    


def filediag():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])#user picked file path
    submit(sttv2.main(file_path),file_path)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(rf"{os.getcwd()}\assets")
def relative_to_assets(path: str) -> Path:return ASSETS_PATH / Path(path)
window = Tk()
window.geometry("1920x1080")
window.configure(bg = "#FFFFFF")
canvas = Canvas(window,bg = "#FFFFFF",height = 1080,width = 1920,bd = 0,highlightthickness = 0,relief = "ridge")
canvas.place(x = 0, y = 0)
canvas.create_rectangle(0.0,0.0,1920.0,1080.0,fill="#1D7A85",outline="")
Label(window,text='PROJECT BY YASH NARAYAN AND PRANAV NARANG',bg='#1D7A85',font=fonts(35),wraplength=400).place(relx=o(260),rely=p(80))#change to heading
Label(window,text="Ask A Question",bg='#1D7A85',fg='#000000',font=fonts(45)).place(relx=o(1235),rely=p(65))
entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_image_1= ImageTk.PhotoImage((Image.open(fr("entry_1.png")).resize((500,80))))
Label(window,image=entry_image_1,bd=0).place(relx=o(1150),rely=p(200),width=(500),height=(80))
entry_1 = Text(bd=0,bg="#FFFFFF",fg="#000716",highlightthickness=0,wrap="word",font=fonts(21))
entry_1.place(relx=o(1200.0),rely=p(220),relwidth=p(300),relheight=o(110.0))
button_image_1 = ImageTk.PhotoImage((Image.open(fr("button_4.png")).resize((t(545),t(130)))))
button1=Button(image=button_image_1,borderwidth=0,highlightthickness=0,command=filediag,relief="flat")
button1.place(relx=o(1175.0),rely=p(700.0),width=(t(545.0)),height=(t(130.0)))
button_image_2 = ImageTk.PhotoImage((Image.open(fr("button_2.png")).resize((t(200),t(72)))))
button_2 = Button(image=button_image_2,borderwidth=0,highlightthickness=0,command=lambda: window.destroy(),relief="flat")
button_2.place(relx=o(1654.0),rely=p(979.0),width=t(200.0),height=t(70))
sih_image=ImageTk.PhotoImage((Image.open(fr("sih.png")).resize((t(800),t(450)))))
Label(window,image=sih_image,bd=0).place(relx=o(30),rely=p(500),height=t(450),width=t(800))
window.resizable(False, False)
window.attributes('-fullscreen',True)
window.bind('<Escape>',lambda x:window.destroy())
entry_1.bind("<Return>", enter)
window.mainloop()
