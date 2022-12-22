import cv2,os
face_classifier = cv2.CascadeClassifier(
    'C:/Users/mdibr/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    global cf
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        cf = img[y:y + h, x:x + w]
    return cf

name = input('enter ur name :')
if os.path.exists('D:/facedata set' + name):
    print('data already present...')
else:
    os.mkdir(r'D:/facedata set/' + name)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = 'D:/facedata set/' + name + '/' + name + '.' + str(
            count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('faces', face)
        else:
            print('face not found')
            pass
        if count == 70:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('done collecting samples..')