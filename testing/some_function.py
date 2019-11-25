from PIL import Image, ImageTk
import tkinter as tk
from tkinter import CENTER
import ScalingAttack
from ScalingAttack2 import create_attack_image, create_attack_image2

# Method for creating attack image in GUI is defined in separate file


def some_function(source_img, target_img, canvas):
    result = Image.new('RGB', (source_img.width + target_img.width, min(source_img.height, target_img.height)))
    result.paste(source_img, (0, 0))
    result.paste(target_img, (source_img.width, 0))
    display_img = ImageTk.PhotoImage(result)
    panel = tk.Label(canvas, image=display_img)
    panel.image = display_img
    canvas.create_window(300, 300, anchor=CENTER, window=panel)


# Can potentially add different scaling functions 
def display_attack_img(source_img, target_img, canvas):
    # This is a pil image
    attack_img = create_attack_image(source_img, target_img, Image.BILINEAR)
    display_img = ImageTk.PhotoImage(attack_img)
    panel = tk.Label(canvas, image=display_img)
    panel.image = display_img
    canvas.create_window(300, 300, anchor=CENTER, window=panel)

def testing_attack_img(source_img, canvas):
    # This is a pil image
    attack_img = create_attack_image2(source_img)
    display_img = ImageTk.PhotoImage(attack_img)
    panel = tk.Label(canvas, image=display_img)
    panel.image = display_img
    canvas.create_window(300, 300, anchor=CENTER, window=panel)



