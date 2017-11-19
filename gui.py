import tkinter
from tkinter import ttk
import os
import subprocess
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.font import Font

import LogoIdentificationWithoutMetadata as fun
from PIL import Image, ImageTk
import cv2

global root
root = Tk()
root.title("IdentifyThat")
root.geometry("1200x600+200+200")
#root.configure(bg = "white")

global photoImg
global labelImg
global labelText

defaultPhoto = "/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/default.jpg"


def showImage(img):
    global photoImg
    global labelImg
    w, h = img.size
    myRatio = w / h

    if myRatio < width / height:
        img = img.resize((int(height * (myRatio)), height), Image.ANTIALIAS)

    else:
        myRatio = h / w
        img = img.resize((width, (int(width * (myRatio)))), Image.ANTIALIAS)

    photoImg = ImageTk.PhotoImage(img)
    labelImg.img = photoImg
    labelImg.config(image=photoImg)
    labelImg.place(relx = 0.004, rely = 0.001, width=width, height=height)


def callback():
    global labelText
    filename =  askopenfilename()
    img, name, prob = fun.findAndIdentify(filename)
    toPrint = ""
    if prob is not "":
        toPrint = '{:04.2f}'.format((prob * 100))

    cv2.imwrite("tempImage.jpg", img)
    img = Image.open("tempImage.jpg")
    showImage(img)

    out = name + "\n" + toPrint if prob == "" else name + "\n" + toPrint+"%"
    labelText.config(text = out, font=("Helvetica", 14))
    labelText.place(relx=0.93, rely=0.5, anchor="c")


button = tk.Button(root, text= "Browse", bg="white")
button.place(relx = .93 , rely = .9 , anchor = "c")
button.config(command = callback, width = 5, height = 2)

height = int(385*1.5)
width = int(675*1.5)

frame = ttk.Frame(root)

frame.place(relx = 0.01, rely=0.02, width=width, height = height)
# button.invoke()
labelImg = tk.Label(frame)
labelText = tk.Label(root)

img = Image.open(defaultPhoto)
showImage(img)


root.mainloop()
