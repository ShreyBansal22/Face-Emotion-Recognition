import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D
from keras.layers import Flatten,MaxPooling2D,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
train_dir='F:/RAJEEV RATAN/18. Deep Surveillance Build a Facial Emotion, Age & Gender Recognition System/fer2013/train'
validation_dir='F:/RAJEEV RATAN/18. Deep Surveillance Build a Facial Emotion, Age & Gender Recognition System/fer2013/validation'
img_rows,img_cols=(48,48)
batch_size=32
num_classes=5
#data augumentaion
train_datagen=ImageDataGenerator(rotation_range=30,width_shift_range=0.4,
                                 height_shift_range=0.4,shear_range=0.3,
                                 fill_mode='nearest',horizontal_flip=True
                                 ,rescale=1./255,)
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(img_rows,img_cols)
                                                  ,color_mode='grayscale'
                                                  ,shuffle=True,batch_size=batch_size,
                                                  class_mode='categorical')
validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(img_rows,img_cols)
                                                  ,color_mode='grayscale'
                                                  ,shuffle=True,batch_size=batch_size,
                                            class_mode='categorical')
#building the small vgg
model=Sequential()
#Adding the layers(32,32)
#layer1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#layer 2 conv-maxpool-dropout(64,64)
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#layer 3 conv-maxpool-dropout(128,128)
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#layer 3 conv-maxpool-dropout(256,256)
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',
                 ))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#first FC layer(64)
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
#second FC layer
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#laast layer
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
 

print(model.summary())


from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
checkpoint=ModelCheckpoint('F:/RAJEEV RATAN/18. Deep Surveillance Build a Facial Emotion, Age & Gender Recognition System/emotion_little_vgg.h5'
                           ,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
earlystop=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks=[checkpoint,earlystop,reduce_lr]

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
nb_train_samples=24176
nb_validation_samples=3006
epochs=30
history=model.fit_generator(train_generator,steps_per_epoch=nb_train_samples//batch_size,epochs=epochs
                    ,verbose=1,callbacks=callbacks,validation_data=validation_generator,
                    validation_steps=nb_validation_samples//batch_size)
print(history.history)
    
import pickle
pickle_out=open("FAce_history.pickle","wb")
pickle.dump(history.history,pickle_out)
pickle_out.close()

pickle_in=open("FAce_history.pickle","rb")    
saved_history=pickle.load(pickle_in)
print(saved_history) 

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

nb_train_samples = 24176
nb_validation_samples = 3006

# We need to recreate our validation generator with shuffle = false
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
#print(validation_generator.class_indices)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
#print(class_labels)
classes = list(class_labels.values())
#print(classes)
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
#print(validation_generator.classes)
print('Classification Report')
target_names = list(class_labels.values())

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)



from keras.models import load_model
from keras.preprocessing.image import img_to_array
classifier=load_model('F:/RAJEEV RATAN/18. Deep Surveillance Build a Facial Emotion, Age & Gender Recognition System/emotion_little_vgg.h5')
import cv2
import numpy as np
 
face_classifier=cv2.CascadeClassifier('C:/Users/S_H_R_E_Y/Desktop/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,6)
    for(x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2) 
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]!=0):
            roi=roi_gray.astype('float')/255.0 
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)   
            preds=classifier.predict(roi)[0]
            label=classes[preds.argmax()]
            label_position=(x,y)
            cv2.putText(gray,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        else:
            cv2.putText(frame,'NO face found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    cv2.imshow("emotion",gray)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()
 



                                               