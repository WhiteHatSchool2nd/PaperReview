
데이터 전처리
===============


### 데이터 셋 읽어오기


```python
#Read dataset
df=pd.read_csv('data/Car_Hacking_5%.csv')
```


9개의 데이터로 구성
CAN ID, DATA0, DATA1, DATA2 ... DATA7



### 데이터 변환

```python
# Transform all features into the scale of [0,1]
# 수치형 특징만 추출
numeric_features = df.dtypes[df.dtypes != 'object'].index

scaler = QuantileTransformer()
#데이터를 [0, 1]범위로 스케일링
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(

    lambda x: (x*255))
```

각 데이터를 스케일링한 결과
![[Pasted image 20240627190427.png]]

### 데이터프레임에서 특정 레이블을 가진 데이터 추출

```python
df0 = df[df['Label']=='R'].drop(['Label'], axis=1) df1 = df[df['Label']=='RPM'].drop(['Label'], axis=1) df2 = df[df['Label']=='gear'].drop(['Label'], axis=1) df3 = df[df['Label']=='DoS'].drop(['Label'], axis=1) df4 = df[df['Label']=='Fuzzy'].drop(['Label'], axis=1)
```


###  각 레이블마다 컬러 이미지 생성 (나머지 레이블에 대해서 동일하게 수행)

```python
# Generate 9*9 color images for class 0 (Normal)
count = 0
ims = []

image_path = "train/0/"
os.makedirs(image_path)

#데이터프레임을 순회하며 이미지 생성 및 저장
for i in range(0, len(df0)):  
    count = count + 1
    if count <= 27: 
        im = df0.iloc[i].values
        ims = np.append(ims, im)
    else:
        ims = np.array(ims).reshape(9, 9, 3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path + str(i) + '.png')
        count = 0
        ims = []
```

### 학습 및 검증 디렉토리 설정

```python
#학습 및 검증 디렉토리 설정
Train_Dir = './train/'
Val_Dir = './test/'
allimgs = []

#학습 디렉토리의 이미지 파일 경로 수
for subdir in os.listdir(Train_Dir):
    for filename in os.listdir(os.path.join(Train_Dir, subdir)):
        filepath = os.path.join(Train_Dir, subdir, filename)
        allimgs.append(filepath)
print(len(allimgs)) # Print the total number of images

#split a test set from the dataset, train/test size = 80%/20%

#전체 이미지 파일 개수의 20%를 검증 세트로 사용
Numbers=len(allimgs)//5     #size of test set (20%)

  
#파일 이동 함수 정의
def mymovefile(srcfile,dstfile):

    if not os.path.isfile(srcfile):

        print ("%s not exist!"%(srcfile))

    else:

        fpath,fname=os.path.split(dstfile)    

        if not os.path.exists(fpath):

            os.makedirs(fpath)              

        shutil.move(srcfile,dstfile)          

        #print ("move %s -> %s"%(srcfile,dstfile))
```


### 검증 세트 생성

```python
# Create the test set
val_imgs = random.sample(allimgs, Numbers)
for img in val_imgs:
    dest_path = img.replace(Train_Dir, Val_Dir)
    mymovefile(img, dest_path)
print('Finish creating test set')

```


### 이미지 리사이즈 함수

```python
# Create the test set
# resize the images 224*224 for better CNN training
def get_224(folder, dstdir):
    imgfilepaths = []
    for root, dirs, imgs in os.walk(folder):
        for thisimg in imgs:
            thisimg_path = os.path.join(root, thisimg)
            imgfilepaths.append(thisimg_path)
    for thisimg_path in imgfilepaths:
        dir_name, filename = os.path.split(thisimg_path)
        dir_name = dir_name.replace(folder, dstdir)
        new_file_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = cv2.imread(thisimg_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(new_file_path, img)
    print('Finish resizing'.format(folder=folder))

#학습 및 검증 이미지 리사이즈즈
DATA_DIR_224 = './train_224/'
get_224(folder='./train/', dstdir=DATA_DIR_224)

DATA_DIR2_224 = './test_224/'
get_224(folder='./test/', dstdir=DATA_DIR2_224)



```
