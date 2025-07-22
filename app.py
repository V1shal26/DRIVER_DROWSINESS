from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import imutils
import dlib
from scipy.spatial import distance
from pygame import mixer
import face_recognition
import os
import time

app = Flask(__name__)

mixer.init()
mixer.music.load("static/assets/music.wav")

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

thresh = 0.25
alert_duration = 5
alert_start_time = None

known_face_encodings = []
known_face_data = {}

image_dir = "major_face/"
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            user_name = filename.split('.')[0]
            known_face_encodings.append(encoding)
            known_face_data[user_name] = {
                "name": user_name
            }
        except IndexError:
            print(f"Warning: No face found in {filename}")

current_user = {"status": "No Face Detected", "user_info": None, "image_path": None}
output_dir = "static/detected_faces"
os.makedirs(output_dir, exist_ok=True)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    global current_user, alert_start_time
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        detected = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_face_data.keys())[first_match_index]
                detected = True
                face_image = frame[top:bottom, left:right]
                face_image_path = os.path.join(output_dir, f"{name}.jpg")
                cv2.imwrite(face_image_path, face_image)

                current_user = {
                    "status": "Face Recognized",
                    "user_info": known_face_data[name],
                    "image_path": f"/static/detected_faces/{name}.jpg"
                }

            else:
                current_user = {"status": "Unknown Face", "user_info": "New User", "image_path": None}

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for subject in subjects:
            shape = predict(gray, subject)
            leftEye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            rightEye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < thresh:
                if alert_start_time is None:
                    alert_start_time = time.time()
                elif time.time() - alert_start_time >= alert_duration:
                    cv2.putText(frame, "***** ALERT *****", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                alert_start_time = None
                if mixer.music.get_busy():
                    mixer.music.stop()

        if not detected:
            current_user = {"status": "No Face Detected", "user_info": "New User", "image_path": None}

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user_info')
def user_info():
    return jsonify(current_user)

@app.route('/static/detected_faces/<filename>')
def get_face_image(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
