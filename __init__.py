from tkinter.ttk import *
from tkinter import Tk, BOTH, W, N, E, S, ALL, CENTER, Frame, Canvas, filedialog
from PIL import Image, ImageTk
from ScalingAttack import create_attack_image_command

class GUI(Frame):

    def __init__(self):
        super().__init__()
        self.create_gui()

    def create_gui(self):

        # This function selects the source image from a selected file
        # Displays the image through a label on the selected canvas
        # Finally, stores the selected pil image in the global source_img variable
        def select_src():
            global source_img
            filename = filedialog.askopenfilename()
            pil_image = Image.open(filename)
            display_image = ImageTk.PhotoImage(pil_image)

            display_lbl = Label(src_img_canvas, image=display_image)
            display_lbl.image = display_image
            src_img_canvas.create_window(300, 133, anchor=CENTER, window=display_lbl)
            source_img = pil_image

        # This function selects the target image from a selected file
        # Displays the image through a label on the selected canvas
        # Finally, stores the selected pil image in the global target_img variable
        def select_target():
            # gets the dir w/ file as file name #
            global target_img
            filename = filedialog.askopenfilename()
            pil_image = Image.open(filename)
            display_image = ImageTk.PhotoImage(pil_image)

            display_lbl = Label(target_img_canvas, image=display_image)
            display_lbl.image = display_image
            target_img_canvas.create_window(300, 133, anchor=CENTER, window=display_lbl)
            target_img = pil_image

        # This function clears the images from the canvas/areas
        def clear(canvas):
            canvas.delete(ALL)

        # GUI construction begins here!
        self.master.title("Scaling Attack Image Generator")
        self.pack(fill=BOTH, expand=True)

        # This allows the modules to scale with window resize
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # Source Image Area
        src_img_canvas = Canvas(self, bg='#FFFFFF')
        src_img_canvas.grid(row=0, column=0, rowspan=4,
                            padx=5, pady=5, sticky=W + N + S + E)
        src_img_lbl = Label(self, text="Source")
        src_img_lbl.grid(row=4, column=0, sticky=W, pady=4, padx=5)

        # Target Image Area
        target_img_canvas = Canvas(self, bg='#FFFFFF')
        target_img_canvas.grid(row=5, column=0, rowspan=4,
                               padx=5, pady=5, sticky=W + N + S + E)
        target_img_lbl = Label(self, text="Target")
        target_img_lbl.grid(row=9, column=0, sticky=W, pady=4, padx=5)

        # Attack Image Area
        attack_img_canvas = Canvas(self, bg='#FFFFFF')
        attack_img_canvas.grid(row=0, column=1, rowspan=9,
                               padx=5, pady=5, sticky=W + N + S + E)
        attack_img_lbl = Label(self, text="Attack")
        attack_img_lbl.grid(sticky=W, row=9, column=1, pady=4, padx=5)

        # Buttons
        choose_source = Button(self, text="Choose Source", command=select_src)
        choose_source.grid(row=10, column=0, sticky=W, padx=5)

        choose_target = Button(self, text="Choose Target", command=select_target)
        choose_target.grid(row=11, column=0, sticky=W, padx=5)

        generate_attack = Button(self, text="Generate", command=lambda: create_attack_image_command(source_img, target_img,
                                                                                                    Image.BILINEAR, attack_img_canvas))
        generate_attack.grid(row=10, column=1, sticky=W, padx=5)

        clear_attack_img = Button(self, text="Clear Attack", command=lambda: clear(attack_img_canvas))
        clear_attack_img.grid(row=10, column=1, sticky=E, padx=5)

        clear_source_img = Button(self, text="Clear Source", command=lambda: clear(src_img_canvas))
        clear_source_img.grid(row=10, column=0, sticky=E, padx=5)

        clear_target_img = Button(self, text="Clear Target", command=lambda: clear(target_img_canvas))
        clear_target_img.grid(row=11, column=0, sticky=E, padx=5)


def main():
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry('%dx%d+0+0' % (width, height))
    app = GUI()
    root.state('zoomed')
    root.mainloop()


if __name__ == '__main__':
    main()

