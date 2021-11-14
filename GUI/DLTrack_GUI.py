"""Python module to create GUI for DL_Track"""

from tkinter import StringVar, Tk, N, S, W, E
from tkinter import ttk, filedialog
from tkinter.tix import *
from threading import Thread, Lock
from PIL import Image
from calculate_architecture import calculateBatch


class DLTrack:
    """Class which provides the utility of a graphical user interface.

    Attributes:
        input_dir: Path to root directory containings all files.
        apo_modelpath: Path to the keras segmentation model.
        fasc_modelpath: Path to the keras segmentation model.
        spacing: Distance (mm) between two scaling lines on ultrasound images.
        scaling: Scanning modaltity of ultrasound image.
        apo_threshold: Threshold for aponeurosis detection.
        fasc_threshold: Threshold for fascicle detection.
        fasc_cont_threshol: Threshould for fascile contours,
        min_width: Minimal allowed width between aponeuroses (mm),
        curvature: Determined fascicle curvature,
        min_pennation: Minimal allowed pennation angle (°),
        max_pennation: Maximal allowed penntaoin angle (°).

    Examples:
        >>>
    """
    def __init__(self, root):

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        root.title("DLTrack")
        root.iconbitmap("home_im.ico")

        main = ttk.Frame(root, padding="10 10 12 12")
        main.grid(column=0, row=0, sticky=(N, S, W, E))
        # Configure resizing of user interface
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.columnconfigure(3, weight=1)
        main.columnconfigure(4, weight=1)
        main.columnconfigure(5, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background = 'DarkSeaGreen3')
        style.configure('TLabel', font=('Lucida Sans', 12),
                        foreground = 'black', background = 'DarkSeaGreen3')
        style.configure('TRadiobutton', background = 'DarkSeaGreen3',
                        foreground = 'black', font = ('Lucida Sans', 12))
        style.configure('TButton', background = 'papaya whip',
                        foreground = 'black', font = ('Lucida Sans', 11))
        style.configure('TEntry', font = ('Lucida Sans', 12), background = 'papaya whip',
                        foregrund = 'black')
        style.configure('TCombobox', background = 'sea green', foreground = 'black')

        # Tooltips
        tip = Balloon(root)
        tip.config(bg="HotPink3", bd=3)
        tip.label.config(bg="linen", fg="black")
        tip.message.config(bg="linen", fg="black", font=("Lucida Sans", 10))
        for sub in tip.subwidgets_all():
            sub.configure(bg='linen')

        # Paths
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        self.input.set("Desktop/DL_Track/")
        # Apo Model path
        self.apo_model = StringVar()
        apo_model_entry = ttk.Entry(main, width=30, textvariable=self.apo_model)
        apo_model_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        self.apo_model.set("Desktop/DL_Track/models/model-apo2-nc.h5")
        # Fasc Model path
        self.fasc_model = StringVar()
        fasc_model_entry = ttk.Entry(main, width=30, textvariable=self.fasc_model)
        fasc_model_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
        self.fasc_model.set("Desktop/DL_Track/models/model-fascSnippets2-nc.h5")
        # Flip File path
        self.flipflag = StringVar()
        flipflag_entry = ttk.Entry(main, width=30, textvariable=self.flipflag)
        flipflag_entry.grid(column=2, row=5, columnspan=3, sticky=(W,E))
        self.flipflag.set("Desktop/DL_Track/FlipFlags.txt")

        #Radiobuttons
        # Image Type
        self.scaling = StringVar()
        static = ttk.Radiobutton(main, text="Bar", variable=self.scaling,
                                 value="Bar")
        static.grid(column=2, row=8, sticky=(W, E))
        manual = ttk.Radiobutton(main, text="Manual", variable=self.scaling,
                                 value="Manual")
        manual.grid(column=3, row=8, sticky=E)
        tip.bind_widget(static,
                        balloonmsg="Choose image type." +
                        " If image taken in static B-mode, choose Static." +
                        " If image taken in other modality, choose Manual" +
                        " in order to scale the image manually.")
        self.scaling.set("Bar")

        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = ("/**/*.tif", "/**/*.tiff", "/**/*.png", "/**/*.bmp",
                    "/**/*.jpeg", "/**/*.jpg")
        filetype_entry = ttk.Combobox(main, width=10, textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=7, sticky=(W,E))
        tip.bind_widget(filetype_entry,
                        balloonmsg="Specifiy filetype of images in root" +
                        " that are taken as whole quadriceps images." +
                        " These images are being prepared for model prediction.")
        self.filetype.set("/**/*.tiff")
        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(main, width=10, textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=9, sticky=(W, E))
        tip.bind_widget(spacing_entry,
                        balloonmsg="Choose disance between scaling bars" +
                                   " in image form dropdown list. " +
                                   "Distance needs to be similar " +
                                   "in all analyzed images.")
        self.spacing.set(10)
        # Apo threshold
        self.apo_threshold = StringVar()
        athresh = (0.1, 0.3, 0.5, 0.7, 0.9)
        apo_entry = ttk.Combobox(main, width=10, textvariable=self.apo_threshold)
        apo_entry["values"] = athresh
        apo_entry.grid(column=2, row=12, sticky=(W,E))
        tip.bind_widget(apo_entry,
                        balloonmsg="Choose or enter threshold used" +
                                    " for aponeurosis prediction.")
        self.apo_threshold.set(0.8)
        # Fasc threshold
        self.fasc_threshold = StringVar()
        fthresh = [0.1, 0.3, 0.5]
        fasc_entry = ttk.Combobox(main, width=10, textvariable=self.fasc_threshold)
        fasc_entry["values"] = fthresh
        fasc_entry.grid(column=2, row=13, sticky=(W,E))
        tip.bind_widget(fasc_entry,
                        balloonmsg="Choose or enter threshold used" +
                                    " for fascicle prediction.")
        self.fasc_threshold.set(0.1)
        # Fasc cont threshold
        self.fasc_cont_threshold = StringVar()
        fcthresh = (20, 30, 40, 50, 60, 70, 80)
        fasc_cont_entry = ttk.Combobox(main, width=10, textvariable=self.fasc_cont_threshold)
        fasc_cont_entry["values"] = fcthresh
        fasc_cont_entry.grid(column=2, row=14, sticky=(W,E))
        tip.bind_widget(fasc_cont_entry,
                        balloonmsg="Choose or enter threshold used" +
                                    " for fascicle contour detection.")
        self.fasc_cont_threshold.set(40)
        # Minimal width
        self.min_width = StringVar()
        mwidth = (20, 30, 40, 50, 60, 70, 80, 90, 100)
        width_entry = ttk.Combobox(main, width=10, textvariable=self.min_width)
        width_entry["values"] = mwidth
        width_entry.grid(column=2, row=15, sticky=(W,E))
        tip.bind_widget(width_entry,
                        balloonmsg="Choose or enter minimal allowed" +
                                    " width between aponeurosis.")
        self.min_width.set(60)
        # Curvature
        self.curvature = StringVar()
        curv = (1,2,3)
        curvature_entry = ttk.Combobox(main, width=10, textvariable=self.curvature)
        curvature_entry["values"] = curv
        curvature_entry["state"] = "readonly"
        curvature_entry.grid(column=2, row=16, sticky=(W,E))
        tip.bind_widget(curvature_entry,
                        balloonmsg="Choose or enter curvature value used" +
                                    " for fascicle detection." +
                                    " The higher the curvature of the fascicles," +
                                    " the higher the curvature value")
        self.curvature.set(1)
        # Minimal pennation
        self.min_pennation = StringVar()
        min_pennation_entry = ttk.Combobox(main, width=10, textvariable=self.min_pennation)
        min_pennation_entry.grid(column=2, row=17, sticky=(W,E))
        tip.bind_widget(min_pennation_entry,
                        balloonmsg="Enter minimal allowed pennation angle.")
        self.min_pennation.set(10)
        # Maximal pennation
        self.max_pennation = StringVar()
        max_pennation_entry = ttk.Combobox(main, width=10, textvariable=self.max_pennation)
        max_pennation_entry.grid(column=2, row=18, sticky=(W,E))
        tip.bind_widget(max_pennation_entry,
                        balloonmsg="Enter maximal allowed pennation angle.")
        self.max_pennation.set(30)

        # Buttons
        # Input directory
        input_button = ttk.Button(main, text="Input",
                                  command=self.get_root_dir)
        input_button.grid(column=5, row=2, sticky=E)
        tip.bind_widget(input_button,
                        balloonmsg="Choose root directory." +
                        " This is the folder containing all images.")
        # Apo model path
        apo_model_button = ttk.Button(main, text="Apo Model",
                                  command=self.get_apo_model_path)
        apo_model_button.grid(column=5, row=3, sticky=E)
        tip.bind_widget(apo_model_button,
                        balloonmsg="Choose apo model path." +
                        " This is the path to the aponeurosis model.")
        # Fasc model path
        fasc_model_button = ttk.Button(main, text="Fasc Model",
                                     command=self.get_fasc_model_path)
        fasc_model_button.grid(column=5, row=4, sticky=E)
        tip.bind_widget(fasc_model_button,
                        balloonmsg="Choose fasc model path." +
                        " This is the path to the fascicle model.")
        # Flipfile model path
        flipfile_button = ttk.Button(main, text="Flip Flags",
                                     command=self.get_flipfile_path)
        flipfile_button.grid(column=5, row=5, sticky=E)
        tip.bind_widget(flipfile_button,
                        balloonmsg="Choose flipfile path." +
                        " This is the path to the file containing the flipflags.")

        # Break Button
        break_button = ttk.Button(main, text="Break", command=self.do_break)
        break_button.grid(column=1, row=19, sticky=W)
        # Run Button
        run_button = ttk.Button(main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=19, sticky=(W, E))

        # Labels
        ttk.Label(main, text="Directories",font=('Verdana', 14)).grid(column=1, row=1, sticky=W)
        ttk.Label(main, text="Root Directory").grid(column=1, row=2)
        ttk.Label(main, text="Apo Model Path").grid(column=1, row=3)
        ttk.Label(main, text="Fasc Model Path").grid(column=1, row=4)
        ttk.Label(main, text="Flip File Path").grid(column=1, row=5)
        ttk.Label(main, text="Image Properties", font=('Verdana', 14)).grid(column=1, row=6,
                  sticky=W)
        ttk.Label(main, text="Image Type").grid(column=1, row=7)
        ttk.Label(main, text="Scaling Type").grid(column=1, row=8)
        ttk.Label(main, text="Spacing (mm)").grid(column=1, row=9)
        ttk.Label(main, text="Analysis Parameters", font=("Verdana", 14)).grid(column=1, row=11,
                sticky=W)
        ttk.Label(main, text="Apo Threshold").grid(column=1, row=12)
        ttk.Label(main, text="Fasc Threshold").grid(column=1, row=13)
        ttk.Label(main, text="Fasc Cont Threshold").grid(column=1, row=14)
        ttk.Label(main, text="Minimal Width").grid(column=1, row=15)
        ttk.Label(main, text="Curvature").grid(column=1, row=16)
        ttk.Label(main, text="Minimal Pennation").grid(column=1, row=17)
        ttk.Label(main, text="Maximal Pennation").grid(column=1, row=18)

        for child in main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # depth_entry.focus()
        root.bind("<Return>", self.run_code)  # execute by pressing return

    def get_root_dir(self):
        """ Asks the user to select the root directory.
            Can have up to two sub-levels.
            All images files (of the same type) in root are analysed.
        """
        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_apo_model_path(self):
        """ Asks the user to select the apo model path.
        """
        apo_model_dir = filedialog.askopenfilename()
        self.apo_model.set(apo_model_dir)
        return apo_model_dir

    def get_fasc_model_path(self):
        """Asks the user to select the fasc model path.
        """
        fasc_model_dir = filedialog.askopenfilename()
        self.fasc_model.set(fasc_model_dir)
        return fasc_model_dir

    def get_flipfile_path(self):
        """Asks the user to selecte the flipflag file.
        """
        flipflag_dir = filedialog.askopenfilename()
        self.flipflag.set(flipflag_dir)
        return flipflag_dir

    def run_code(self):
        """ The code is run upon clicking.
        """

        if self.is_running:
            # don't run again if it is already running
            return
        self.is_running = True

        selected_input_dir = self.input.get()
        selected_apo_model_path = self.apo_model.get()
        selected_fasc_model_path = self.fasc_model.get()
        selected_flipflag_path = self.flipflag.get()
        selected_filetype = self.filetype.get()
        selected_scaling = self.scaling.get()
        selected_spacing = self.spacing.get()
        selected_apo_threshold = self.apo_threshold.get()
        selected_fasc_threshold = self.fasc_threshold.get()
        selected_fasc_cont_threshold = self.fasc_cont_threshold.get()
        selected_min_width = self.min_width.get()
        selected_curvature = self.curvature.get()
        selected_min_pennation = self.min_pennation.get()
        selected_max_pennation = self.max_pennation.get()

        thread = Thread(
            target=calculateBatch,
            args=(
                selected_input_dir,
                selected_apo_model_path,
                selected_fasc_model_path,
                selected_flipflag_path,
                selected_filetype,
                selected_scaling,
                selected_spacing,
                selected_apo_threshold,
                selected_fasc_threshold,
                selected_fasc_cont_threshold,
                selected_min_width,
                selected_curvature,
                selected_min_pennation,
                selected_max_pennation,
                self,
            ))

        thread.start()

    @property
    def should_stop(self):
        self._lock.acquire()
        should_stop = self._should_stop
        self._lock.release()
        return should_stop

    @property
    def is_running(self):
        self._lock.acquire()
        is_running = self._is_running
        self._lock.release()
        return is_running

    @should_stop.setter
    def should_stop(self, flag: bool):
        self._lock.acquire()
        self._should_stop = flag
        self._lock.release()

    @is_running.setter
    def is_running(self, flag: bool):
        self._lock.acquire()
        self._is_running = flag
        self._lock.release()

    def do_break(self):
        if self.is_running:
            self.should_stop = True


if __name__ == "__main__":
    root = Tk()
    DLTrack(root)
    root.mainloop()
