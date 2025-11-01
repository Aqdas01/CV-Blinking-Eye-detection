import cv2
import cvzone
from cvzone.PlotModule import LivePlot
from cvzone.FaceMeshModule import FaceMeshDetector
cap = cv2.VideoCapture('videos/bandicam 2025-10-24 20-24-28-124.mp4')
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[20,50])
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList=[]
BlinkCounter = 0
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

        leftUp = face[159]
        leftdown = face[23]
        leftLeft = face[130]
        leftRight =face[243] 
        lengthVer ,_  = detector.findDistance(leftUp, leftdown)
        lenghtHor , _=detector.findDistance(leftLeft,leftRight) 
        
        
        cv2.line(img, leftUp, leftdown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        
        ratio = int((lengthVer/lenghtHor)*100)
        ratioList.append(ratio)
        if len(ratioList)>3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)
        if ratioAvg<35:
            BlinkCounter +=1
            cvzone.putTextRect(img, f'Blink Count :{BlinkCounter}', (100,100) )
        
        
        
        imgPlot = plotY.update(ratioAvg)
        img = cv2.resize(img, (740, 550))
        imgStack =  cvzone.stackImages([img,imgPlot],2, 1)
    
    else:
        img = cv2.resize(img, (640, 550))
        imgStack =  cvzone.stackImages([img,img],2, 1)
    
    cv2.imshow('image', imgStack)
    
    cv2.waitKey(25)
