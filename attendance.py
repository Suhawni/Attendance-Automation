# attendance.py

import face_recognition
import cv2 as cv
import numpy as np
import csv
import os
from datetime import datetime

def mark_attendance():
    video_capture = cv.VideoCapture(0)
    video_capture.set(3, 640)  # Set width
    video_capture.set(4, 480)  # Set height
    a = face_recognition.load_image_file("1.jpg")
    a_enc = face_recognition.face_encodings(a)[0]

    b = face_recognition.load_image_file("2.jpg")
    b_enc = face_recognition.face_encodings(b)[0]

    known_face_enc = [a_enc, b_enc]
    known_face_names = ["Einstein", "Selena"]

    students = known_face_names.copy()
    face_names = []

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    csv_file_path = os.path.join(
        'C:\\Users\\hp\\OneDrive\\Desktop\\face', f'{current_date}.csv')

    # Create a dictionary to track if a student has already been marked present
    student_present = {name: False for name in students}

    # Define 'f' outside the try block
    f = None

    try:
        f = open(csv_file_path, 'w+', newline='')
        lnwriter = csv.writer(f)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Couldn't read frame. Exiting...")
                break

            x = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = x[:, :, ::-1]
            face_loc = face_recognition.face_locations(rgb_small)
            face_enc = face_recognition.face_encodings(rgb_small, face_loc)
            face_names = []

            for (top, right, bottom, left), enc in zip(face_loc, face_enc):
                match = face_recognition.compare_faces(known_face_enc, enc)
                name = ""
                face_dis = face_recognition.face_distance(known_face_enc, enc)
                best_match = np.argmin(face_dis)

                if match[best_match]:
                    name = known_face_names[best_match]
                    if not student_present[name]:
                        print(name, ": Present")
                        students.remove(name)
                        student_present[name] = True
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, current_time])

                # Draw a rectangle around the detected face
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                face_names.append(name)

            cv.imshow("Attendance Automation System", frame)
            key = cv.waitKey(1)
            if key & 0xFF == ord('m'):  # Break the loop when 'q' is pressed
                break
    except Exception as e:
        print("Error:", str(e))
    finally:
        video_capture.release()
        cv.destroyAllWindows()
        if f is not None:
            f.close()
