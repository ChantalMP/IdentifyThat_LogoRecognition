import tkinter
from tkinter import ttk
import os
import subprocess
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
import LogoIdentificationWithoutMetadata as fun
from PIL import Image, ImageTk
import cv2

global root
root = Tk()
root.title("First GUi")
root.geometry("800x400+200+200")
#root.configure(bg = "white")

global photoImg
global panelImg
global panelText

defaultPhoto = "/media/psf/Google Drive/Python Projects/RohdeSchwarzHackatum/default.jpg"


def showImage(img):
    global photoImg
    global panelImg
    w, h = img.size
    myRatio = w / h

    if myRatio < width / height:
        img = img.resize((int(height * (myRatio)), height), Image.ANTIALIAS)

    else:
        myRatio = h / w
        img = img.resize((width, (int(width * (myRatio)))), Image.ANTIALIAS)

    photoImg = ImageTk.PhotoImage(img)
    panelImg.img = photoImg
    panelImg.config(image=photoImg)
    panelImg.place(relx = 0.05, rely = 0.05, width=width, height=height)


def callback():
    global panelText
    filename =  askopenfilename()
    img, name = fun.findAndIdentify(filename)
    cv2.imwrite("tempImage.jpg", img)
    img = Image.open("tempImage.jpg")
    showImage(img)


    panelText.config(text = name)
    panelText.place(relx=0.93, rely=0.5, anchor="c")


button = tk.Button(root, text= "Browse", bg="white")
button.place(relx = .93 , rely = .9 , anchor = "c")
button.config(command = callback, width = 5, height = 2)

height = 385
width = 675

frame = ttk.Frame(root)

frame.place(relx = 0.01, rely=0.02, width=width, height = height)
# button.invoke()
panelImg = tk.Label(frame)
panelText = tk.Label(root)

img = Image.open(defaultPhoto)
showImage(img)


root.mainloop()
