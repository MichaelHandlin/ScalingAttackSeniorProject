import tkinter as tk
from tkinter import filedialog
from scaling import *
from PIL import Image


class GUI(object):
    CANVAS_HEIGHT = 400
    CANVAS_WIDTH = 600
    HEADER_WIDTH = 600
    HEADER_HEIGHT = 50

    F_WIDTH = 300
    F_HEIGHT = 250
    new_size = (200, 200)

    def __init__(self, canvas):
        self.canvas = canvas
        self.create_gui()

       #self.image = image from function that uses button to get pic

    def create_gui(self):
        self.canvas = tk.Canvas(height=self.CANVAS_HEIGHT, width=self.CANVAS_WIDTH, bg='#FFFFFF')
        self.canvas.grid() # fix so that entire gui aligns with window

        # basic format
        self.header = tk.Frame(self.canvas, bg='#000000', height=self.HEADER_HEIGHT, width=self.HEADER_WIDTH)
        self.header.grid(row=0)

        self.f1 = tk.Frame(self.canvas, bg='#8C8C8C', height=self.F_HEIGHT, width=self.F_WIDTH)
        self.f1.grid(row=1, column=0, sticky=tk.W)

        self.select_img_btn = tk.Button(self.f1, text='Choose target image', command=self.select_file)
        self.select_img_btn.pack(fill='none', expand=True) # fix this to align button
        self.input_image = Image.open(self.filename)




        self.f2 = tk.Frame(self.canvas, bg='#F8CD93', height=self.F_HEIGHT, width=self.F_WIDTH)
        self.f2.grid(row=1, column=0, sticky=tk.E)

    def select_file(self):
        self.filename = filedialog.askopenfilename()

    def scale_image(self):
        self.resized_img = self.image.resize(self.new_size, resample=Image.NEAREST)
        self.resized_img.show()


root = tk.Tk()
gui = GUI(root)
root.mainloop()

