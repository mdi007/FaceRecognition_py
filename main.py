import os,math, cv2, numpy as np
from os import listdir
from os.path import join, isfile

face_classifier = cv2.CascadeClassifier(
    'C:/Users/mdibr/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def facedetect(img):
    global roi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is (True):
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# -----------

onlypicsall = []
dataPath0 = 'D:/facedata set/'
folders = os.listdir(r'D:/facedata set/')
for i in range(len(folders)):
    dataPath = 'D:/facedata set/' + folders[i] + '/'
    onlypics = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    onlypicsall.extend(onlypics)
training_data = []
labels = []
cc=0
for i in range(len(folders)):
    for j in range(70):
        imgPath = dataPath0 + '/' + folders[i] + '/' + onlypicsall[cc]
        cc+=1
        images = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
labels = list(range(len(folders) * 70))
labels = np.asarray(labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data), np.asarray(labels))
# print('done train')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = facedetect(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        print(result)
        yo=math.floor(result[0]/70)
        cv2.putText(image, folders[yo], (250,450), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('faceRecognition', image)
    except:
        cv2.putText(image, "face not found", (250, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('faceRecognition', image)
        pass
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
