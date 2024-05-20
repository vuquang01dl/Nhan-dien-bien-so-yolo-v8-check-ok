import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO
from PIL import Image, ImageTk


class ANPRWithButton:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.text = ""
        self.frame = None  # Initialize frame
        self.root = tk.Tk()
        self.root.title("ANPR")
        self.root.geometry("800x600")  # Set window size
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.load_button = tk.Button(self.frame, text="Tải ảnh lên", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.recognize_button = tk.Button(self.frame, text="Nhận dạng", command=self.process_frame)
        self.recognize_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.frame, text="Lưu ảnh", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(self.frame, text="Thoát", command=self.root.destroy)
        self.quit_button.pack(side=tk.RIGHT)

        self.text_label = tk.Label(self.root, text="Biển số xe: ")
        self.text_label.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.model = YOLO("best.pt")  # Load the YOLO model

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.frame = cv2.imread(file_path)  # Save frame to class attribute
            self.display_image()

    def process_frame(self):
        frame = self.frame  # Use the saved frame
        if frame is not None:
            results = self.model.predict(source=frame)
            
            for result in results:
                boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
                confidences = result.boxes.conf  # Confidence scores
                
                for box, confidence in zip(boxes, confidences):
                    if confidence > 0.25:  # Ensure the detection is confident enough
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Crop the detected license plate area
                        cropped_plate = frame[y1:y2, x1:x2]
                        
                        # Use EasyOCR to read text from the cropped license plate
                        text_results = self.reader.readtext(cropped_plate)
                        
                        # Extract and print the detected text
                        if text_results:
                            combined_text = " ".join([res[1] for res in text_results])
                            self.text = re.sub(r'[^A-Z0-9]', '', combined_text.upper())
                            formatted_text = self.format_license(self.text)
                            self.text_label.config(text="Biển số xe: " + formatted_text)

                            # Draw the bounding box and text on the image
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_size = cv2.getTextSize(formatted_text, font, 1, 2)[0]
                            text_x = int((x1 + x2 - text_size[0]) / 2)
                            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            frame = cv2.putText(frame, formatted_text, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            self.frame = frame
            self.display_image()

    def display_image(self):
        if self.frame is not None:
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if file_path:
            cv2.imwrite(file_path, self.frame)

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        if len(text) == 7:
            text += ' '  # Add 0 if the length of text is 7

        if len(text) < 8:
            return ""

        license_plate_ = ''
        dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'L': '4'}
        dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 3: dict_char_to_int, 4: dict_char_to_int,
                   5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

anpr_with_button = ANPRWithButton()
anpr_with_button.root.mainloop()
