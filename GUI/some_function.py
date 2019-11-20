from PIL import Image, ImageTk
import tkinter as tk
from tkinter import CENTER

# Method for creating attack image in GUI is defined in separate file


def some_function(source_img, target_img, canvas):
    result = Image.new('RGB', (source_img.width + target_img.width, min(source_img.height, target_img.height)))
    result.paste(source_img, (0, 0))
    result.paste(target_img, (source_img.width, 0))
    display_img = ImageTk.PhotoImage(result)
    panel = tk.Label(canvas, image=display_img)
    panel.image = display_img
    canvas.create_window(300, 300, anchor=CENTER, window=panel)


