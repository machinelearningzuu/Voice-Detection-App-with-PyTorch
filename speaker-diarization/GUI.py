from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from threading import Thread
import pyaudio
import wave
import math
from datetime import datetime

from inference import *
from pdf_generation import *

window = Tk()

window.title("VOICE RECORDER")

window.minsize(width=800, height=600)
window.config(padx=5, pady=5)

Recording_ON =False
timer = None

def reset_contdown():
    global timer
    window.after_cancel(timer)


def count_down(count):
    count_minutes = math.floor(count/60)
    count_sec = count%60
    
    if count_sec<10:
        count_sec = f"0{count_sec}"
    
    time_string=f'{count_minutes}:{count_sec}'
    Timer_Label.config(text=time_string)  
    global timer
    timer=window.after(1000,count_down,count+1)

def Record_sound():
    global audio
    global farmes
    global stream
    
    global Recording_ON
    Recording_ON =True
    
    Rec_Label['text']="Recording"
    count_down(0)
    
    Stop_Btn['state'] = tk.NORMAL
    Start_Btn['state'] = tk.DISABLED
    
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = chunk)

    farmes = []

    while Recording_ON:
        data = stream.read(1024)
        farmes.append(data)


def start_thread():
    global Recording_ON
    Recording_ON = True
    
    
    # Create and launch a thread 
    t = Thread (target = Record_sound)
    t.start()


def Stop_Rec():
    global Recording_ON
    global stream
    reset_contdown()
    
    
    SaveRec_Btn['state'] = tk.NORMAL
    Stop_Btn['state'] = tk.DISABLED
    Start_Btn['state'] = tk.NORMAL
    Rec_Label['text']="Recording Stopped"
    
    Recording_ON =False
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
def Save_Rec():
    SaveRec_Btn['state'] = tk.DISABLED
    Stop_Btn['state'] = tk.DISABLED
    Start_Btn['state'] = tk.NORMAL

    now = datetime.now()
    DT=now.strftime("%d_%m_%Y-%H.%M.%S")

    sound_file = wave.open(save_file_name.format(f"Rec_{DT}.wav"),"wb")
    
    
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b''.join(farmes))
    sound_file.close()

def UploadAction(event=None):
    wavFile = filedialog.askopenfilename()
    file_name = os.path.split(wavFile)[-1]
    window.destroy()

    speakerdf, summarydf = run(wavFile)
    tbl_contents = format_speaker_outputs(speakerdf)
    tb2_contents = format_speaker_summary(summarydf)
    pdfGeneration(tbl_contents, tb2_contents, file_name)

if __name__ == '__main__':
    Rec_Label = Label(font=("Arial", 20, "bold"))
    Rec_Label.place(x=300, y=500)

    Timer_Label = Label(font=("Arial", 20, "bold"))
    Timer_Label.place(x=205, y=500)

    Upload_Btn = Button(
                        font=("Arial", 20, "bold"), 
                        text='UPLOAD', 
                        command=UploadAction
                        )
    Upload_Btn.config(padx=15,pady=5)
    Upload_Btn.place(x=280, y=150)

    Start_Btn = Button(text="START", font=("Arial", 20, "bold"),
                    command=start_thread)
    Start_Btn.config(padx=15,pady=5)
    Start_Btn.place(x=100, y=400)

    Stop_Btn = Button(text="STOP", font=("Arial", 20, "bold"),
                    command=Stop_Rec,
                    state=tk.DISABLED)
    Stop_Btn.config(padx=15,pady=5)
    Stop_Btn.place(x=300, y=400)

    SaveRec_Btn = Button(text="SAVE REC", font=("Arial", 20, "bold"),
                        command=Save_Rec,
                        state=tk.DISABLED)
    SaveRec_Btn.config(padx=15,pady=5)
    SaveRec_Btn.place(x=500, y=400)


    TITLE_Label = Label(font=("Arial", 40, "bold"))
    TITLE_Label.place(x=100, y=10)
    TITLE_Label['text']="Speaker Diarization Tool"

    VS_Label = Label(font=("Arial", 30, "bold"))
    VS_Label.place(x=330, y=280)
    VS_Label['text']="OR"

    window.mainloop()