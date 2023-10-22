import os
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
fonts = lambda a:("Inter ExtraBold", a * -1,'bold')
o=lambda x:x/1920
p=lambda x:x/1080
fr = lambda x: rf"{os.getcwd()}\build\assets\frame0\{x}"
t=lambda s:9*s//10
def dest():
    for i in ['sb','msgbox']:
        try:
            exec(f'{i}.destroy()')
        except:pass
def enter(event):
    # Check if the Shift key is also pressed (Shift+Enter)
    if event.keysym == 'Return' and not event.state & 0x0001:
        submit()
        return "break"
    else:
        entry_1.event_generate('<<Return>>')
def submit():
    dest()
    inp = (entry_1.get(1.0, "end-1c"))
    output='heloo'#output recieved from chatbot
    msgbox = Text(window,wrap='word',font=fonts(24))
    msgbox.insert(END,output)
    msgbox.place(relx=o(1165),rely=p(330),relheight=p(300),relwidth=o(585))
    sb = Scrollbar(window,orient= "vertical")
    sb.config(command=msgbox.yview)
    sb.place(relx=o(1760),rely=p(330),relheight = p(300))
def sub():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])#audio file's path
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(rf"{os.getcwd()}\build\assets\frame0")
def relative_to_assets(path: str) -> Path:return ASSETS_PATH / Path(path)
window = Tk()
window.geometry("1920x1080")
window.configure(bg = "#FFFFFF")
canvas = Canvas(window,bg = "#FFFFFF",height = 1080,width = 1920,bd = 0,highlightthickness = 0,relief = "ridge")
canvas.place(x = 0, y = 0)
canvas.create_rectangle(0.0,0.0,1920.0,1080.0,fill="#1D7A85",outline="")
Label(window,text="Ask A Question",bg='#1D7A85',fg='#000000',font=fonts(45)).place(relx=o(1235),rely=p(65))
entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_image_1= ImageTk.PhotoImage((Image.open(fr("entry_1.png")).resize((500,80))))
Label(window,image=entry_image_1,bd=0).place(relx=o(1150),rely=p(200))
entry_1 = Text(bd=0,bg="#FFFFFF",fg="#000716",highlightthickness=0,wrap="word",font=fonts(21))
entry_1.place(relx=o(1200.0),rely=p(220),relwidth=p(300),relheight=o(110.0))
button_image_1 = ImageTk.PhotoImage((Image.open(fr("button_1.png")).resize((t(545),t(130)))))
button1=Button(image=button_image_1,borderwidth=0,highlightthickness=0,command=sub,relief="flat")
button1.place(relx=o(1175.0),rely=p(700.0),relwidth=o(595.0),relheight=p(141.0))
button_image_2 = ImageTk.PhotoImage((Image.open(fr("button_2.png")).resize((200,70))))
button_2 = Button(image=button_image_2,borderwidth=0,highlightthickness=0,command=lambda: window.destroy(),relief="flat")
button_2.place(relx=o(1654.0),rely=p(979.0),relwidth=o(241.0),relheight=p(83.28125))
sih_image=ImageTk.PhotoImage((Image.open(fr("sih.png")).resize((800,450))))
Label(window,image=sih_image,bd=0).place(relx=o(30),rely=p(500))
window.resizable(False, False)
window.attributes('-fullscreen',True)
window.bind('<Escape>',lambda x:window.destroy())
entry_1.bind("<Return>", enter)
window.mainloop()
