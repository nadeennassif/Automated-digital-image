import tkinter
import customtkinter
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.util import random_noise
from skimage import data
from numpy.fft import fft
import cv2
import numpy as np

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

from matplotlib import pyplot as plt


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        # configure window
        self.title("Image Project")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create left frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Image Project",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Upload",hover_color = 'white',fg_color = 'white',text_color = "black",
                                                        command=self.Upload_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Fourir transform",
                                                        command=self.fourir_transform)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Sobel and Laplace",
                                                        command=self.SobelAndLaplace)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))


        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Histogram")
        self.tabview.add("periodic noise")
        self.tabview.add("salt&paper")

        self.tabview.tab("Histogram").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("periodic noise").grid_columnconfigure(0, weight=1)
        self.tabview.tab("salt&paper").grid_columnconfigure(0, weight=1)


        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Histogram calculate",
                                                           command=self.Calculate_histogram)
        self.string_input_button.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Histogram Equalization",
                                                           command=self.Histogram_Equalizatioon)
        self.string_input_button.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.string_input_button = customtkinter.CTkButton(self.tabview.tab("Histogram"), text="Plot",
                                                           command=self.Plot_Fun)
        self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"),
                                                   text="Add noise",
                                                   command=self.add_peridic_Noise)
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"), text="Notch/Band-reject",
                                                   command=self.notch)
        self.label_tab_2.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("periodic noise"), text="Mask",
                                                   command=self.mask_filter)
        self.label_tab_2.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Add",
                                                   command=self.add_salt_paper_noise)
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Remove",
                                                   command=self.remove_salt_and_pepper)
        self.label_tab_2.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.label_tab_2 = customtkinter.CTkButton(self.tabview.tab("salt&paper"), text="Plot",
                                                   command=self.Plot_Fun)
        self.label_tab_2.grid(row=2, column=0, padx=20, pady=(10, 10))

        # create right frame

        # self.radiobutton_frame = customtkinter.CTkFrame(self)
        # self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # set default values

        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        self.textbox.insert("0.0", "Creating a GUI that allows the user to:\n"
                                   "- Upload an image\n- Calculate its histogram and display it\n"
                                   "- Apply histogram equalization and display both equalized image and its histogram\n"
                                   "- Apply filtering (Sobel, Laplace) + user types parameters and display them\n"
                                   "- Apply Fourier Transform of image and display it\n"
                                   "- Add noise (Salt and pepper, Periodic) + user types parameters and display noisy image\n"
                                   "- Remove S&P using median + user types parameters and display clean image\n"
                                   "- Remove periodic noise (user selects method: Notch/Band-reject/Mask), for mask method user is allowed to select 2 pixels on Fourier Transform of noisy image display for you to remove it.\n"
                                   "- Notch/band reject: you detect and remove noise AUTOMATICALLY. User will NOT give any coordinates. Then you remove noise.\n"
                                   "- Mask: user only SELECTS 2 pixels. User will NOT give any coordinates. Just SELECTS with a MOUSE CLICK.Then you remove noise.")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def Calculate_histogram(self):
        hist = np.histogram(image, bins=np.arange(0, 255))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        ax1.axis('off')
        ax2.stem(hist[1][:-1], hist[0])
        ax2.set_title('Histogram of grey values')
        plt.show()
        return hist

    def sidebar_button_event(self):
        print("sidebar_button click")

    def Upload_image(self):
        file_path = tkinter.filedialog.askopenfilename()
        global image
        global image2
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return image

    def apply_filter(self,noisy_fourier, filter_i):
        # this function apply the filter on the fourier of the noisy image and returns the filtered image in frequency domain
        # and spatial domain

        # Input:
        #      1-fourier of the noisy image
        #      2-the filter

        fshift = noisy_fourier * filter_i
        filtered_img_ft = np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return filtered_img_ft, img_back
        # Helper Function

        ########################################## function for gui #######################################
        # takes the noisy image, opens a pop-up window with the fourier of the noisy image, the user chooses 2 pixels to zero out, and then we zero out their 12 pixels neighbours
        # returns the filtered image in frequency and spatial domain

    def mask_filter(self):
        global noisy_image
        file_path = tkinter.filedialog.askopenfilename()
        noisy_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        global x1_coord
        global y1_coord
        global x2_coord
        global y2_coord

        # First convert the clear image to frquency domain
        dft_clear = cv2.dft(np.float32(noisy_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift_clear = np.fft.fftshift(dft_clear)

        magnitude_spectrum = np.log(cv2.magnitude(dft_shift_clear[:, :, 0], dft_shift_clear[:, :, 1]) + 1)
        # Showing the the image in frequency domain
        plt.figure(figsize=(7, 5));
        plt.imshow(magnitude_spectrum, cmap='gray');
        # plt.title('Input Image in Frequency Domain'), plt.xticks([]), plt.yticks([]);
        plt.xticks([]), plt.yticks([]);
        # saving the image
        plt.savefig('image in frequency domain.png', bbox_inches='tight', pad_inches=0)
        plt.show()

        dim = magnitude_spectrum.shape
        # print('dim',dim)
        img = cv2.imread('image in frequency domain.png')
        dim = (dim[1], dim[0])
        # print('dim',dim)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        # print(img.shape)

        x_array = []
        y_array = []

        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (x, y)
                x_array.append(y)
                y_array.append(x)
                cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
                # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                #  1.0, (0, 0, 0), thickness=1)
                # cv2.imshow("image", img)
                # print(x,y)

        while True:
            # Display the image to the user
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 800, 800)
            cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            cv2.imshow('image', img)

            # Wait for a key press
            key = cv2.waitKey(1) & 0xFF

            # If the user has selected two pixels, break out of the loop
            if (len(x_array) and len(y_array) == 2):
                # coordinates of the first pixel
                x1_coord = int(x_array[len(x_array) - 2])
                y1_coord = int(y_array[len(y_array) - 2])
                # coordinates of the second pixel
                x2_coord = int(x_array[len(x_array) - 1])
                y2_coord = int(y_array[len(y_array) - 1])
                break
        # Close all windows
        cv2.destroyAllWindows()

        # removing the noise by mask
        image_forier = dft_shift_clear;
        # Creating a filter with only zeros at the periodic noise frequencies
        PN_filter = np.ones(dft_shift_clear.shape)

        x1_coord = x1_coord + 2
        y1_coord = y1_coord - 3
        x2_coord = x2_coord + 2
        y2_coord = y2_coord - 3
        # removing 12 pixels around the choosed pixels to make sure of removing the periodic noise perfectly
        PN_filter[x1_coord - 3:x1_coord + 3,
        y1_coord - 3:y1_coord + 3] = 0  ## here i should replace those with the x and y coordinates
        PN_filter[x2_coord - 3:x2_coord + 3,
        y2_coord - 3:y2_coord + 3] = 0  ##//////////////////////////////////////////////
        # print(PN_filter.sum())# should be less

        # apply mask and inverse DFT
        filtered_img_ft, img_back = self.apply_filter(dft_shift_clear, PN_filter)
        plt.imshow(filtered_img_ft, cmap='gray')
        plt.show()
        plt.imshow(img_back, cmap='gray')
        plt.show()
        return filtered_img_ft, img_back

    def Remove_periodic_noise_Notch(self):
        print("")

    def add_periodic_noise(self):
        shape = image.shape[0], image.shape[1]
        noise = np.zeros(shape, dtype='float64')

        x, y = np.meshgrid(range(0, shape[0]), range(0, shape[1]))
        s = 1 + np.sin(x + y / 1.5)
        noisy_periodic_Image = ((image) / 128 + s) / 4

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
        ax1.imshow(image, 'gray')
        ax2.imshow(noisy_periodic_Image, 'gray')

        plt.show()

    def add_salt_paper_noise(self):
        global noise_Salt_Paper_img
        noise_Salt_Paper_img = random_noise(image, mode='s&p', amount=0.3)

        noise_Salt_Paper_img = np.array(255 * noise_Salt_Paper_img, dtype='uint8')

        # Display the noise image
        plt.imshow(noise_Salt_Paper_img, 'gray')
        plt.show()
        cv2.waitKey(0)

    def remove_salt_and_pepper(self):
        # Take Input from user
        dialog = customtkinter.CTkInputDialog(text="Remove nose by mean or median: 1:median, 2:mean:", title="Test")
        noise_x = int(dialog.get_input())
        if (noise_x == 1):
            # Take Input from user
            doalogDisk_x = customtkinter.CTkInputDialog(text="input Parameter of Disk:", title="Test")
            Disk_x = int(doalogDisk_x.get_input())
            ImgNewwithoutnoise = median(noise_Salt_Paper_img, disk(Disk_x))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(16, 8)
            ax1.imshow(noise_Salt_Paper_img, 'gray')
            ax2.imshow(ImgNewwithoutnoise, 'gray')
            plt.show()
            ## denoise image
        elif (noise_x == 2):
            # mean filter (average)
            m = 5
            n = 5
            ImgNewwithoutnoise_mean = cv2.blur(noise_Salt_Paper_img, (m, n))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(16, 8)
            ax1.imshow(noise_Salt_Paper_img, 'gray')
            ax2.imshow(ImgNewwithoutnoise_mean, 'gray')
            plt.show()


    def Histogram_Equalizatioon(self):
        dialoginput_x = customtkinter.CTkInputDialog(text="Histogram Equalization please choose one of types and input:1:histogram equalization (global) ,2:CLAHE (local) Adaptive histogram equalization:",
                                                    title="Test")
        input_x = int(dialoginput_x.get_input())
        if (input_x == 1):
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show()

            hist, bins = np.histogram(image.flatten(),
                                      bins=256,
                                      range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.hist(image.flatten(),
                    bins=256,
                    range=[0, 256],
                    color='r')
            ax.set_xlabel('pixel intensity')
            ax.set_ylabel('#pixels')
            ax.set_xlim(0, 255)

            ax2 = ax.twinx()
            ax2.plot(cdf_normalized, color='b')
            ax2.set_ylabel('cdf')
            ax2.set_ylim(0, 1)

            plt.show()

            equ = cv2.equalizeHist(image)
            plt.imshow(equ, cmap='gray')
            plt.axis('off')
            plt.show()

            hist, bins = np.histogram(equ.flatten(),
                                      bins=256,
                                      range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.hist(equ.flatten(),
                    bins=256,
                    range=[0, 256],
                    color='r')
            ax.set_xlabel('pixel intensity')
            ax.set_ylabel('#pixels')
            ax.set_xlim(0, 255)

            ax2 = ax.twinx()
            ax2.plot(cdf_normalized, color='b')
            ax2.set_ylabel('cdf')
            ax2.set_ylim(0, 1)

            plt.show()
        elif (input_x == 2):

            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show()

            hist, bins = np.histogram(image.flatten(),
                                      bins=256,
                                      range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.hist(image.flatten(),
                    bins=256,
                    range=[0, 256],
                    color='r')
            ax.set_xlabel('pixel intensity')
            ax.set_ylabel('#pixels')
            ax.set_xlim(0, 255)

            ax2 = ax.twinx()
            ax2.plot(cdf_normalized, color='b')
            ax2.set_ylabel('cdf')
            ax2.set_ylim(0, 1)

            plt.show()
            # CLAHE (local) Adaptive histogram equalization (AHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahef = clahe.apply(image)
            plt.imshow(clahef, cmap='gray')
            plt.axis('off')
            plt.show()

            hist, bins = np.histogram(clahef.flatten(),
                                      bins=256,
                                      range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.hist(clahef.flatten(),
                    bins=256,
                    range=[0, 256],
                    color='r')
            ax.set_xlabel('pixel intensity')
            ax.set_ylabel('#pixels')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 60000)

            ax2 = ax.twinx()
            ax2.plot(cdf_normalized, color='b')
            ax2.set_ylabel('cdf')
            ax2.set_ylim(0, 1)

            plt.show()
        elif (input_x == 3):
            def piece_wise(img, r1, s1, r2, s2):
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if (0 <= img[i][j] and img[i][j] <= r1):
                            img[i][j] = (s1 / r1) * img[i][j]
                        elif (r1 < img[i][j] and img[i][j] <= r2):
                            img[i][j] = ((s2 - s1) / (r2 - r1)) * (img[i][j] - r1) + s1
                        else:
                            img[i][j] = ((255 - s2) / (255 - r2)) * (img[i][j] - r2) + s2

                    return img

            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show()

            hist, bins = np.histogram(image.flatten(),
                                      bins=256,
                                      range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()

            fig, ax = plt.subplots(figsize=(5, 5))

            ax.hist(image.flatten(),
                    bins=256,
                    range=[0, 256],
                    color='r')
            ax.set_xlabel('pixel intensity')
            ax.set_ylabel('#pixels')
            ax.set_xlim(0, 255)

            ax2 = ax.twinx()
            ax2.plot(cdf_normalized, color='b')
            ax2.set_ylabel('cdf')
            ax2.set_ylim(0, 1)

            plt.show()
            plt.imshow(piece_wise(data.camera(), 70, 30, 140, 225), 'gray', vmin=0, vmax=255)
        else:
            print("  In valid Please Enter one of the choises")


    def Plot_Fun(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        ax1.imshow(image2, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax2.imshow(image, cmap=plt.cm.gray)
        plt.show()

    def SobelAndLaplace(self):
        img = image

        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

        lap = np.uint8(np.absolute(lap))

        sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)

        sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

        sobelX = np.uint8(np.absolute(sobelX))

        sobelY = np.uint8(np.absolute(sobelY))

        sobelCombined = cv2.bitwise_or(sobelX, sobelY)

        titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined']

        images = [img, lap, sobelX, sobelY, sobelCombined]

        for i in range(5):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')

            plt.title(titles[i])

            plt.xticks([]), plt.yticks([])  # ticks

        plt.show()

    # Filter: Low pass filter
    def fourir_transform(self):


        # open the image f

        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

        # transform the image into frequency domain, f --> F
        F = np.fft.fft2(image)
        Fshift = np.fft.fftshift(F)

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(F)), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
        plt.axis('off')
        plt.show()

        # Create Gaussin Filter: Low Pass Filter
        M, N = image.shape
        H = np.zeros((M, N), dtype=np.float32)
        D0 = 10
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u - M / 2) * 2 + (v - N / 2) * 2)
                H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))

        plt.figure(figsize=(5, 5))
        plt.imshow(H, cmap='gray')
        plt.axis('off')
        plt.show()

        # Image Filters
        Gshift = Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))

        plt.figure(figsize=(5, 5))
        plt.imshow(g, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(G)), cmap='gray')
        plt.axis('off')
        plt.show()

        # Gaussian: High pass filter
        HPF = 1 - H

        plt.figure(figsize=(5, 5))
        plt.imshow(HPF, cmap='gray')
        plt.axis('off')
        plt.show()

        # Image Filters
        Gshift = Fshift * HPF
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))

        plt.figure(figsize=(5, 5))
        plt.imshow(g, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log1p(np.abs(G)), cmap='gray')
        plt.axis('off')
        plt.show()

    def add_peridic_Noise(self, orientation = "vertical"):
        frequency = 2
        img = image.copy() / 255  # .astype('float32')

        if orientation == "vertical":

            for n in range(img.shape[1]):
                if n % frequency:
                    img[:, n] += np.sin(.1 * np.pi * n)

        elif orientation == "horizontal":

            for n in range(img.shape[0]):
                if n % frequency:
                    img[n, :] += np.sin(.1 * np.pi * n)

        img *= 255
        img = np.uint8(np.clip(img, 0, 255))
        plt.title('After adding noise')
        plt.imshow(img, cmap='gray')
        plt.show()

    def notch(self):
        image_array = np.array(image)
        # Fourier Transform
        fourier_transform = np.fft.fftshift(np.fft.fft2(image_array))

        # Size of Image
        m = np.shape(fourier_transform)[0]
        n = np.shape(fourier_transform)[1]

        u = np.arange(m)
        v = np.arange(n)

        # Find the center
        u0 = int(m / 2)
        v0 = int(n / 2)

        # Bandwidth
        D0 = 10

        notch = np.zeros(np.shape(fourier_transform))

        for x in u:
            for y in v:
                D1 = np.sqrt((x-m/2-u0)**2 + (y-n/2-v0)**2)
                D2 = np.sqrt((x - m / 2 + u0) ** 2 + (y - n / 2 + v0) ** 2)
                notch[x][y] = 1 - np.exp(-0.5 * D1 * D2 / (D0 ** 2))

        # Apply the filter
        fourier_transform = fourier_transform + notch
        image_array = np.fft.ifft2(np.fft.ifftshift(fourier_transform))
        plt.imshow(image, cmap='gray')
        plt.show()
        return image_array

if __name__ == "__main__":
    app = App()
    app.mainloop()
