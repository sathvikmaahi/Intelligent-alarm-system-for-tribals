import cv2
#import facevec
import numpy as np
#import smtplib
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models  import load_model
from twilio.rest import Client

model = load_model(r'F:\MY PROJECTS\VEC PROJECTS\INTELLIGENT ALERT SYSTEM FOR TRIBAL\Training\alert.h5') 
video = cv2.VideoCapture(0)
name = ['forest','with fire']
    
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg",frame)
    img = image.load_img("image.jpg",target_size = (64,64))
    x  = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    pred=np.argmax(model.predict(x),axis=1)
    #pred = model.predict_classes(x)
    pred=model.predict(x)
    #p = pred[0]
    p=int(pred[0][0])
    print(pred)
    cv2.putText(frame, "predicted  class = "+str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    
    
    pred = model.predict(x)
    pred=np.argmax(model.predict(x),axis=1)
    if pred[0]==1:
        account_sid = 'ACbaedc0f433eb384b9fc9957a506c2b99'
        auth_token = '8d5cd7a614764ea80686686ffb243ca5'
        client = Client(account_sid, auth_token)

        message = client.messages \
        .create(
         body='Forest Fire is detected, stay alert',
         from_='+15133275578', #twilio free number
         to='+919100588408')
        print(message.sid)
    
        print('Fire Detected')
        print ('SMS sent!')
        break
    else:
        print("no danger")
       #break
    cv2.imshow("image",frame)
   
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

video.release()
cv2.destroyAllWindows()