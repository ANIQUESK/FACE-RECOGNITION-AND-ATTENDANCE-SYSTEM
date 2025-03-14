from sys import path
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import os
import mysql.connector
import cv2
import numpy as np
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Train:

    def __init__(self, root):
        self.root = root
        self.root.geometry("1566x829+0+0")
        self.root.title("Train Panel")

        # This part is image labels setting start 
        # first header image  
        img = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\banner.jpg")
        img = img.resize((1566, 150), Image.Resampling.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)

        # set image as label
        f_lb1 = Label(self.root, image=self.photoimg)
        f_lb1.place(x=0, y=0, width=1566, height=150)

        # background image 
        bg1 = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\t_bg1.jpg")
        bg1 = bg1.resize((1566, 799), Image.Resampling.LANCZOS)
        self.photobg1 = ImageTk.PhotoImage(bg1)

        # set image as label
        bg_img = Label(self.root, image=self.photobg1)
        bg_img.place(x=0, y=130, width=1566, height=799)

        # title section
        title_lb1 = Label(bg_img, text="Welcome to Training Panel", font=("verdana", 34, "bold"), bg="white", fg="navyblue")
        title_lb1.place(x=0, y=0, width=1566, height=55)

        # Create buttons below the section 
        # ------------------------------------------------------------------------------------------------------------------- 
        # Training button 1
        std_img_btn = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\t_btn1.png")
        std_img_btn = std_img_btn.resize((200, 200), Image.Resampling.LANCZOS)
        self.std_img1 = ImageTk.PhotoImage(std_img_btn)

        std_b1 = Button(bg_img, command=self.train_classifier, image=self.std_img1, cursor="hand2")
        std_b1.place(x=700, y=200, width=200, height=200)

        std_b1_1 = Button(bg_img, command=self.train_classifier, text="Train Dataset", cursor="hand2", font=("tahoma", 15, "bold"), bg="white", fg="navyblue")
        std_b1_1.place(x=700, y=400, width=200, height=50)

    # ==================Create Function of Training===================
    def train_classifier(self):
        data_dir = "data_img"
        paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        
        faces = []
        ids = []

        for image in paths:
            img = Image.open(image).convert('L')  # convert to gray scale 
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)

            cv2.imshow("Training", imageNp)
            if cv2.waitKey(1) == 13:
                break
        
        ids = np.array(ids)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(faces, ids, test_size=0.2, random_state=42)
        
        #=================Train Classifier=============
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(X_train, y_train)
        
        # Validate the model
        y_pred = []
        for face in X_val:
            pred_id, _ = clf.predict(face)
            y_pred.append(pred_id)
        
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        
        clf.write("clf.xml")

        cv2.destroyAllWindows()
        messagebox.showinfo("Result", f"Training Dataset Completed! Validation Accuracy: {accuracy * 100:.2f}%", parent=self.root)

if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()
    