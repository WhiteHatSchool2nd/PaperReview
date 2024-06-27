### Generate Training and Test Imatges

```python
#generate training and test images

TARGET_SIZE=(224,224)   #이미지의 목표 크기

INPUT_SIZE=(224,224,3)  #모델에 입력될 이미지의 크기 마지막 3은 RGB

BATCHSIZE=128   #could try 128 or 32

  

#Normalization

#ImageDataGenerator는 이미지 데이터를 실시간으로 증강 및 전처리하는 도구

train_datagen = ImageDataGenerator(rescale=1./255)

  

test_datagen = ImageDataGenerator(rescale=1./255)

#rescale1./255는 모든 이미지의 픽셀 값을 0과 1사이로 정규화 픽셀 값을 255로 나눈다.

  

#훈련데이터 생성

train_generator = train_datagen.flow_from_directory(

        './train_224/',

        target_size=TARGET_SIZE,

        batch_size=BATCHSIZE,

        class_mode='categorical')

  

#테스트 데이터 생성

validation_generator = test_datagen.flow_from_directory(

        './test_224/',

        target_size=TARGET_SIZE,

        batch_size=BATCHSIZE,

        class_mode='categorical')
```

### Define the image plotting functions

```python
#plot the figures

class LossHistory(keras.callbacks.Callback):

    #훈련이 시작될 때 loss와 accuracy를 저장할 빈 리스트를 초기화

    def on_train_begin(self, logs={}):

        self.losses = {'batch':[], 'epoch':[]}

        self.accuracy = {'batch':[], 'epoch':[]}

        self.val_loss = {'batch':[], 'epoch':[]}

        self.val_acc = {'batch':[], 'epoch':[]}

    #배치가 끝날 떄 호출되는 콜백 각 배치가 끝날 때 해당 배치의 손실과 정확도를 기록

    def on_batch_end(self, batch, logs={}):

        self.losses['batch'].append(logs.get('loss'))

        self.accuracy['batch'].append(logs.get('acc'))

        self.val_loss['batch'].append(logs.get('val_loss'))

        self.val_acc['batch'].append(logs.get('val_acc'))

  

    # 에포크가 끝날 떄 호출되는 콜백 해당 에포크의 손실과 정확도를 기록

    def on_epoch_end(self, batch, logs={}):

        self.losses['epoch'].append(logs.get('loss'))

        self.accuracy['epoch'].append(logs.get('acc'))

        self.val_loss['epoch'].append(logs.get('val_loss'))

        self.val_acc['epoch'].append(logs.get('val_acc'))

    #손실 및 정확도 시각화

    def loss_plot(self, loss_type):

        iters = range(len(self.losses[loss_type]))

        plt.figure()

        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':

            # acc

            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')

            # loss

            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

            # val_acc

            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')

            # val_loss

            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)

        plt.xlabel(loss_type)

        plt.ylabel('acc-loss')

        plt.legend(loc="upper right")

        plt.show()
```


### CNN model by own

```python
#입력데이터의 형태, 분류할 클래스의 수, 훈련할 에포크 수, 훈련된 모델을 저장할 파일 경로

def cnn_by_own(input_shape,num_class,epochs,savepath='./model_own.h5'):

    model = Sequential()    #시퀀셜 모델 -> 레이어를 순차적으로 쌓는 모델 형태

    #첫 번째 Conv2D 레이어는 64개의 3x3필터를 사용하여 입력 이미지를 컨볼루션

    #두 번째 Conv2D 레이어는 동일한 설정으로 추가 컨볼루션진행

    #MaxPooling2D레이어는 2x2 폴링 윈도우를 사용하여 특성 맵을 다운샘플링

    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=input_shape,padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    #두 개의 Conv2D레이어와 하나의 MaxPolling2D 레이어로 구성 필터 수가 128개로 증가

    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    #세 개의 Conv2D 레이어와 GlobalAveragePooling2D레이어로 구성 필터 수가 256개로 증가

    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))

    model.add(GlobalAveragePooling2D()) #각 특성 맵의 평균을 계산하여 글로벌 피처 벡터를 만든다.

    #완전 연결(Dense)레이어

    #첫 번째 Dense레이어는 256개의 유닛을 가지고 있으며 ReLu활성화 함수를 사용

    #Dropdout래이어는 50%의 드롭아웃 비율을 사용하여 과적합을 방지

    #마지막 Dense레이어는 분류할 클래스 수만큼 유닛을 가지고 있으며 softmax활성화 함수를 사용

    model.add(Dense(256,activation='relu'))

    model.add(Dropout(rate=0.5))

    model.add(Dense(num_class,activation='softmax'))

    #모델 컴파일 손실함수는 categorical_crossentropy, 옵티마이저는 adam, 평가 매트릭으로 accuracy사용

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #train model

    #earlystopping콜백은 검증 정확도가 개선되지 않을 경우 훈련을 조기 종료

    earlyStopping=kcallbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='auto')

    #ModelCheckpoint콜백은 검증 정확도가 최고일 때 모델을 저장

    saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    #fit_generator함수를 사용하여 모델을 훈련

    hist=model.fit_generator(

        train_generator,    #훈련 데이터 제공

        steps_per_epoch=len(train_generator),   #에포크당 스텝 수 설정

        epochs=epochs,  #에포크 수 설정

        validation_data=validation_generator,   #에포크당 스텝 수 설정

        validation_steps=len(validation_generator),

        callbacks=[earlyStopping,saveBestModel,history_this],

    )
```