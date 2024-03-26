from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
from flask import Flask, render_template, Response, send_from_directory

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width
video_capture.set(4, 480)  # Set height

# Load known face encodings and names
a = face_recognition.load_image_file("1.jpg")
a_enc = face_recognition.face_encodings(a)[0]

b = face_recognition.load_image_file("2.jpg")
b_enc = face_recognition.face_encodings(b)[0]

known_face_enc = [a_enc, b_enc]
known_face_names = ["Einstein", "Selena"]

students = known_face_names.copy()
student_present = {name: False for name in students}

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_path = os.path.join(os.getcwd(), f'{current_date}.csv')

try:
    f = open(csv_file_path, 'w+', newline='')
    lnwriter = csv.writer(f)
except Exception as e:
    print("Error:", str(e))
    f = None

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Couldn't read frame. Exiting...")
            break

        x = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
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
                    status="Present"
                    student_present[name] = True
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time,status])
                    print(f"JavaScript: updateStatus('{name}');")
                    
                    # Read the CSV file after writing a row
                    f.flush()  # Ensure that the data is written to the file
                    f.seek(0)  # Move the file pointer to the beginning of the file
                    reader = csv.reader(f)
                    data = list(reader)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            face_names.append(name)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    data = []
    try:
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
    except Exception as e:
        print("Error:", str(e))

    return render_template('index.html', data=data)

@app.route('/video_feed')
def read_camera():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get-csv')
def get_csv():
    return send_from_directory(os.getcwd(), f'{current_date}.csv')

if __name__ == '__main__':
    app.run(debug=True)