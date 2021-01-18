

#EXECUTE : python test.py ou python3 test.py si vouz avez python3 sur votre ordinateur
# ETAPE 1: importer les bibliotheques comme Keras - openCV et numpy
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#ETAPE2: il faut choisir un classificateur et un modele 
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# Charger face-model et puis detecter le visage d'apres 'INPUT-FRAME'
classifier =load_model('./Emotion_Detection.h5')
# Charger 'EMOTION-MODEL'
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
# Preparer les different classes 'angry 'sad' 'happy' 'neutral'.....
# on vas utiliser VideoCapture pour capturer la video
cap = cv2.VideoCapture(0)


#ETAPE3 
while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    #cap.read retourn un valeur boolean(TRUE/FALSE) c'est a dire si le cadre est lue correctement il va retourner TRUE  
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #CV2.cvtColor(input_image, flag) c'est une fonction qu'on peut utiliser pour convertir l'image d'un seul couleur en BGR2GRAY par exemple
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #face_classifier.detectMultiScale c'est pour trouver les visages

#ETAPE4    dessiner un rectangle 
        # x y w h ce sont les coordonnees du visage
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # nous pouvons utiliser n'importe quelle couleur que nous voulons, nous avons juste besoin de changer cette valeur (255,0,0)
        #cv2.rectangle il va retourner une image encadre avec des bordure
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
	#on va changer la dimension du rectangle en 48,48 

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

	# on va convaertir l'image en tableau

#ETAPE5 prédire l'émotion et mettre l'étiquette sur l'image
        # faire une prédiction sur ROI, puis rechercher la classe
	# dans cette ETAPE l'algorithme va chercher les probabilites possible pour detecter les emotions d'un visage

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#Suppression de fenêtres en appelant la fonction ci-dessous destroyAllWindows()
cv2.destroyAllWindows()



























