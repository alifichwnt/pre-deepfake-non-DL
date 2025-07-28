"Tes WebCam"
import cv2
cap = cv2.VideoCapture(0)
print("Webcam OK" if cap.isOpened() else "Webcam Error")
cap.release()

"TES FACESWAP"
import mediapipe as mp
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

target_img = cv2.imread("Target_Pict.jpg") #ganti `Target_Pict` dengan nama file anda
gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("Wajah tidak terdeteksi di foto target!")
    exit()


(x, y, w, h) = faces[0]
face_crop = target_img[y:y+h, x:x+w]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h_frame, w_frame, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]

        landmark_points = [(int(lm.x * w_frame), int(lm.y * h_frame)) for lm in face_landmarks.landmark]
        x_min = min([p[0] for p in landmark_points])
        x_max = max([p[0] for p in landmark_points])
        y_min = min([p[1] for p in landmark_points])
        y_max = max([p[1] for p in landmark_points])

        face_width = x_max - x_min
        face_height = y_max - y_min
        resized_face = cv2.resize(face_crop, (face_width, face_height))
        
        center = (x_min + face_width // 2, y_min + face_height // 2)

        mask = 255 * np.ones(resized_face.shape, resized_face.dtype)

        # Clone wajah target ke frame
        frame = cv2.seamlessClone(resized_face, frame, mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Real-Time Face Swap (Blended)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()