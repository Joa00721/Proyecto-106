import cv2
import numpy as np


# Crear nuestro clasificador de cuerpos
bodyc=cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Inicializar la captura de video para nuestro archivo de video
cap = cv2.VideoCapture('walking.avi')

# Comenzar el bucle una vez que el video esté cargado exitosamente
while True:
    
    # Leer el primer cuadro
    ret, frame = cap.read()

    # Convertir cada cuadro a escala de grises
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pasar el cuadro a nuestro clasificador de cuerpos
    bodies = bodyc.detectMultiScale(gray, 1.2, 3)
    
   
    for(x,y,w,h)in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(30,87,98),15)
        cv2.imshow("peatones",frame)

    if cv2.waitKey(1) == 32: #32 es la barra espaciadora
        break

cap.release()
cv2.destroyAllWindows()
