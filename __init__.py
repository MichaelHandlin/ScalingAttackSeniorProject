from tkinter.ttk import *
import tkinter as tk
from tkinter import Tk, BOTH, W, N, E, S, ALL, CENTER, Frame, Canvas, filedialog
from PIL import Image, ImageTk
from some_function import some_function


class GUI(Frame):

    def __init__(self):
        super().__init__()
        self.create_GUI()

    def create_GUI(self):

        def select_src():
            # gets the dir w/ file as file name #
            global source_img
            filename = filedialog.askopenfilename()
            pil_image = Image.open(filename)
            display_image = ImageTk.PhotoImage(pil_image)

            panel = Label(src_img_canvas, image=display_image)
            panel.image = display_image
            src_img_canvas.create_window(300, 133, anchor=CENTER, window=panel)
            source_img = pil_image

        def select_target():
            # gets the dir w/ file as file name #
            global target_img
            filename = filedialog.askopenfilename()
            pil_image = Image.open(filename)
            display_image = ImageTk.PhotoImage(pil_image)

            panel = Label(target_img_canvas, image=display_image)
            panel.image = display_image
            target_img_canvas.create_window(300, 133, anchor=CENTER, window=panel)
            target_img = pil_image

        def clear(canvas):
            canvas.delete(ALL)

        def some_function(source_img, target_img):
            result = Image.new('RGB', (source_img.width + target_img.width, min(source_img.height, target_img.height)))
            result.paste(source_img, (0, 0))
            result.paste(target_img, (source_img.width, 0))
            display_img = ImageTk.PhotoImage(result)
            panel = tk.Label(attack_img_canvas, image=display_img)
            panel.image = display_img
            attack_img_canvas.create_window(300, 300, anchor=CENTER, window=panel)

        self.master.title("Scaling Attack Image Generator")
        self.pack(fill=BOTH, expand=True)

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

        #generate_attack = Button(self, text="Generate", command=lambda: scale_img(test_img, (400,200)))
        generate_attack = Button(self, text="Generate", command=lambda: some_function(source_img, target_img))
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