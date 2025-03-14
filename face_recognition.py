from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import mysql.connector
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import csv
import time

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1566x829+0+0")
        self.root.title("Face Recognition Panel")

        # Database configuration
        self.db_config = {
            'user': 'root',
            'password': 'admin',
            'host': 'localhost',
            'database': 'face_recognition',
            'port': 3306
        }

        # Attendance tracking with a 1-hour cooldown
        self.attendance_tracker = {}
        self.attendance_cooldown = timedelta(hours=1)

        # Cache student data to minimize database calls
        self.student_cache = self.load_student_data()

        # UI Configuration
        self.FONT = ("verdana", 30, "bold")
        self.BUTTON_FONT = ("tahoma", 15, "bold")
        self.BG_COLOR = "white"
        self.FG_COLOR = "navyblue"

        # Video capture and recognition settings
        self.video_cap = None
        self.is_running = False
        self.frame_skip = 2  # Process every 2nd frame to reduce load
        self.frame_count = 0

        # Load face recognition model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists("clf.xml"):
            self.recognizer.read("clf.xml")
        else:
            messagebox.showerror("Error", "Classifier file 'clf.xml' not found!")
            self.root.destroy()

        self.setup_ui()

    def load_student_data(self):
        """Load student data from the database to reduce queries."""
        student_cache = {}
        try:
            with mysql.connector.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT Student_ID, Name, Roll_No FROM student")
                    for student_id, name, roll_no in cursor.fetchall():
                        student_cache[student_id] = (name, roll_no)
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error loading student data: {err}")
        return student_cache

    def setup_ui(self):
        """Initialize UI components"""
        try:
            banner_img = Image.open("Images_GUI/banner.jpg").resize((1566, 150), Image.LANCZOS)
            self.photoimg = ImageTk.PhotoImage(banner_img)
            Label(self.root, image=self.photoimg).place(x=0, y=0, width=1566, height=150)

            bg_img = Image.open("Images_GUI/bg2.jpg").resize((1566, 799), Image.LANCZOS)
            self.photobg1 = ImageTk.PhotoImage(bg_img)
            bg_label = Label(self.root, image=self.photobg1)
            bg_label.place(x=0, y=130, width=1566, height=799)

            Label(bg_label, text="Welcome to Face Recognition Panel", 
                 font=self.FONT, bg=self.BG_COLOR, fg=self.FG_COLOR).place(x=0, y=0, width=1566, height=50)

            # Face Detector button
            btn_img = Image.open("Images_GUI/f_det.jpg").resize((200, 200), Image.LANCZOS)
            self.std_img1 = ImageTk.PhotoImage(btn_img)

            Button(bg_label, command=self.start_face_recognition_thread, 
                  image=self.std_img1, cursor="hand2").place(x=700, y=190, width=200, height=200)
            Button(bg_label, command=self.start_face_recognition_thread, text="Face Detector", 
                  cursor="hand2", font=self.BUTTON_FONT, bg=self.BG_COLOR, fg=self.FG_COLOR
                  ).place(x=700, y=390, width=200, height=45)

        except FileNotFoundError as e:
            messagebox.showerror("File Error", f"Required image file not found: {e.filename}")

    def start_face_recognition_thread(self):
        """Start face recognition in a separate thread"""
        if self.is_running:
            messagebox.showinfo("Info", "Face recognition is already running!")
            return

        self.is_running = True
        recognition_thread = threading.Thread(target=self.face_recognition, daemon=True)
        recognition_thread.start()

    def mark_attendance(self, student_id):
        """Record attendance with cooldown validation"""
        current_time = datetime.now()

        if student_id in self.attendance_tracker:
            last_marked = self.attendance_tracker[student_id]
            if current_time - last_marked < self.attendance_cooldown:
                return  # Skip if within cooldown period

        self.attendance_tracker[student_id] = current_time
        name, roll_no = self.student_cache.get(student_id, ("Unknown", "N/A"))

        try:
            with open("attendance.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([student_id, roll_no, name, current_time.strftime("%H:%M:%S"), 
                                 current_time.strftime("%d/%m/%Y"), "Present"])
        except IOError as e:
            messagebox.showerror("File Error", f"Failed to save attendance: {str(e)}")
            
    def face_recognition(self):
        """Main face recognition method optimized for speed"""
        self.video_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution for faster processing
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_cap.set(cv2.CAP_PROP_FPS, 15)

        while self.is_running:
            ret, frame = self.video_cap.read()
            if not ret:
                messagebox.showwarning("Camera Error", "Failed to capture frame")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue  # Skip frames to reduce processing load

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                student_id, confidence = self.recognizer.predict(face_roi)
                confidence_pct = int(100 * (1 - confidence/300))

                if confidence_pct > 85 and student_id in self.student_cache:  # Increased threshold
                    self.draw_face_info(frame, x, y, w, h, student_id)
                    self.mark_attendance(student_id)
                else:
                    self.draw_unknown_face(frame, x, y, w, h)

            cv2.imshow("Face Detector", frame)
            if cv2.waitKey(1) == 13:  # Exit on Enter key
                break

        self.is_running = False
        self.video_cap.release()
        cv2.destroyAllWindows()
        def face_recognition(self):
            """Main face recognition method optimized for speed"""
            self.video_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution for faster processing
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_cap.set(cv2.CAP_PROP_FPS, 15)

            while self.is_running:
                ret, frame = self.video_cap.read()
                if not ret:
                    messagebox.showwarning("Camera Error", "Failed to capture frame")
                    break

                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue  # Skip frames to reduce processing load

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    student_id, confidence = self.recognizer.predict(face_roi)
                    confidence_pct = int(100 * (1 - confidence/300))

                    if confidence_pct > 85 and student_id in self.student_cache:  # Increased threshold
                        self.draw_face_info(frame, x, y, w, h, student_id)
                        self.mark_attendance(student_id)
                    else:
                        self.draw_unknown_face(frame, x, y, w, h)

                cv2.imshow("Face Detector", frame)
                if cv2.waitKey(1) == 13:  # Exit on Enter key
                    break

            self.is_running = False
            self.video_cap.release()
            cv2.destroyAllWindows()

    def draw_face_info(self, frame, x, y, w, h, student_id):
        """Draw recognized face information on the frame"""
        name, _ = self.student_cache.get(student_id, ("Unknown", "N/A"))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(frame, f"ID: {student_id}, Name: {name}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_unknown_face(self, frame, x, y, w, h):
        """Draw unknown face information"""
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def on_closing(self):
        """Handle window closing event"""
        self.is_running = False
        if self.video_cap:
            self.video_cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = Face_Recognition(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()