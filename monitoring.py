from db_connection import *
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
from datetime import datetime
import time

#ini
#Monitoring
def monitoring(nama,stambuk, matkul):
    start_time = datetime.now()
    start_time = start_time.strftime("%D:%H:%M:%S")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    model = load_model("cnn_model2.h5")
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    cap = cv2.VideoCapture(1)
    Score = 0
    tidak_aktif = 0
    waktu_tidak_fokus = 0
    aktif =0
    waktu = 0

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        if isinstance(eyes, tuple):
            waktu = waktu+1
            aktif = 0
            cv2.rectangle(frame, (0, 50), (100, 0), (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(frame, str(waktu), (10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(50, 255, 50), thickness=1, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(frame, 'Tidak Aktif', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                        color=(50, 50, 255),
                        thickness=1, lineType=cv2.LINE_AA)
            tidak_aktif = tidak_aktif + 1
            time.sleep(1)
            if tidak_aktif > 5:
                sound.play()
                if tidak_aktif > 10:
                    waktu_tidak_fokus = waktu_tidak_fokus + 1
                    cv2.rectangle(frame, (0, height - 50), (300, height), (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, 'Tidak Aktif : ' + str(waktu_tidak_fokus) + ' detik', (10, height - 20),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(50, 50, 255),
                                thickness=1, lineType=cv2.LINE_AA)
            else:
              #  tidak_aktif = 0
                sound.stop()
            

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 50, 50), thickness=3)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(50, 255, 50), thickness=2)

            # preprocessing steps
            eye = frame[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            # preprocessing is done now model prediction
            prediction = model.predict(eye)
            print(prediction)
            print(aktif, "detik aktif")
            print(tidak_aktif, "detik tidak aktif")
            # if eyes are closed
            if prediction[0] < 0.98:
                waktu = waktu+1
                cv2.rectangle(frame, (0, 50), (100, 0), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, str(waktu), (10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(50, 255, 50), thickness=1, lineType=cv2.LINE_AA)
                cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, 'Tidak Aktif', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1,
                            color=(50, 50, 255),
                            thickness=1, lineType=cv2.LINE_AA)
                tidak_aktif = tidak_aktif + 1
                aktif = 0
                time.sleep(1)
                if (tidak_aktif > 5):
                    sound.play()
                    if tidak_aktif > 10:
                        waktu_tidak_fokus = waktu_tidak_fokus + 1
                        cv2.rectangle(frame, (0, height - 50), (300, height), (0, 0, 0), thickness=cv2.FILLED)
                        cv2.putText(frame, 'Tidak Aktif : ' + str(waktu_tidak_fokus) + ' detik', (10, height - 20),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                    color=(50, 50, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                else:
                    sound.stop()
                # print("|",waktu,"|\tNilai Prediksi = ", prediction, "\tHasil Prediksi = Tidak Aktif \tKeterangan = Mata Terdeteksi", "\t\tWaktu Tidak Aktif = ", waktu_tidak_fokus)
            # if eyes are open
            elif prediction[0] > 0.98:
                waktu = waktu+1
                cv2.rectangle(frame, (0, 50), (100, 0), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, str(waktu), (10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(50, 255, 50), thickness=1, lineType=cv2.LINE_AA)
                tidak_aktif = 0
                aktif = aktif + 1
                Score = 0
                cv2.rectangle(frame, (0, height - 50), (100, height), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, 'Aktif', (10, height - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                            color=(50, 255, 50),
                            thickness=1, lineType=cv2.LINE_AA)
                time.sleep(1)
                if (aktif > 60):
                    sound.play()
                    if (aktif > 65):
                        waktu_tidak_fokus = waktu_tidak_fokus + 1
                        cv2.rectangle(frame, (0, height - 50), (300, height), (0, 0, 0), thickness=cv2.FILLED)
                        cv2.putText(frame, 'Tidak Aktif : ' + str(waktu_tidak_fokus) + ' detik', (10, height - 20),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                    color=(50, 50, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
                else:
                    sound.stop()
                # print("|",waktu,"|\tNilai Prediksi = ", prediction, "\tHasil Prediksi = Aktif \t\t\tKeterangan = Mata Terdeteksi", "\t\tWaktu Tidak Aktif = ", waktu_tidak_fokus)
        cv2.imshow('Sistem Monitoring Mahasiswa', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            end_time = datetime.now()
            end_time = end_time.strftime("%D:%H:%M:%S")
            db = MySQLdb.connect("bjpm1nw3h1yvq88rsybl-mysql.services.clever-cloud.com", "uvuy3aoryk4jnwx0",
                                 "RYpyOl9JhGEKc3DIY3OM",
                                 "bjpm1nw3h1yvq88rsybl")
            insertrec = db.cursor()
            sqlquery = "INSERT INTO " + str(matkul) +\
                       "(Nama, Stambuk, Mata_Kuliah, Waktu_Masuk, Waktu_Keluar, Waktu_Tidak_Fokus_Detik) " \
                       "VALUES(%s, %s, %s, %s, %s, %s)"
            value = (nama, stambuk, matkul, start_time, end_time, waktu_tidak_fokus)
            insertrec.execute(sqlquery, value)
            db.commit()
            print("===================================================================================================================================")
            print("                                                      HASIL MONITORING                                                             ")
            print("===================================================================================================================================")
            print("Nama :", nama, "\t\tNIM :", stambuk,"\tWaktu Masuk :", start_time, "\tWaktu Keluar :", end_time, "Waktu Tidak Fokus :", waktu_tidak_fokus, "detik")
            print("Sukses mengirim hasil monitoring ke database !!!")
            break

    cap.release()
    cv2.destroyAllWindows()