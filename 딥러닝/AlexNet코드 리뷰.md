# AlexNet코드 리뷰

## maxPoolLayer ()

```python
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)
```

1. **maxPoolLayer()의 하이퍼파라미터들:**  maxpooling을 하는 함수, tf.nn.max_pool()에 들어갈 파라미터 kHeight와 kWidth는 커널 사이즈를 의미하고 strideX와 strideY는 stride의 값을 의미
2. **tf.nn.max_pool() :** 입력에 대해 max pooling을 해줌 max pooling은 학습이 없음(가중치가 없음)
3. **tf.nn.max_pool() 의  하이퍼파라미터들 :** padding = "SAME"은 주위에 패딩을 하여 stride가 1일 경우 최종 output의 크기가 input의 크기와 동일하게 하는 것을 의미  

## dropout()

```python
def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)
```

1. **dropout:**훈련data에서만 좋은 성과를 내는 Overfitting을 막기위해 해주는 것 dropout은 모델에서 몇개의 연결을 끊어서 훈련하는 것으로, 즉 몇개의 노드들을 랜덤하게 죽이고 남은 노드들을 통해서만 훈련한다. 
2. **tf.nn.dropout과 파라미터들**: dropout을 실행해줌 x값으로 fcLayer을 거친 값이 들어옴

## LRN()

```python
def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)
```

1. **LRN:** 같은 위치의 픽셀에 대해서 복수의 feature map간에 정규화를 하는 방법, 옆억제를 시행하는 정규화 방법 지금은 별로 쓰이지 않는 방법 Batch Normalization을 사용.

2. **tf.nn.local_response_normalization의 의미와 파라미터 값**: 

   ```python
   sqr_sum[a, b, c, d] =
       sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
   output = input / (bias + alpha * sqr_sum) ** beta
   ```

   위의 코드와 같이 동작한다. 파라미터들의 의미는 x =  input, R = depth_radius , alpha = alpha , beta=beta , bias = bias

## fcLayer()

```python
def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out
```

 tf.nn.xw_plus_b()는 matmul (x, weights) + bias 계산함 matmul은 곱하는 것임

## convLayer ()

```python
def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convolution"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)
```

코드 리뷰

```python
channel = int(x.get_shape()[-1])은 channel은 rgb값 channel/groups는 필터 깊이 featureNum은 필터 개수
```

```python
 conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding) 는 conv = def function(a,b){ return  tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding) }을 의미한다 tf.nn.conv2d는 Convolution layers를 실행하는 함수로 a는 input b는 filter이다 
```

```python
  with tf.variable_scope(name) as scope : 는 이름을 정해주는 코드이고 
       
  w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])는 필터를 의미
  b = tf.get_variable("b", shape = [featureNum]) bias를 의미한다
```

```python
tf.split은 텐서를 한 차원을 기준으로 여러개의 텐서로 나누는 함수이다.
xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)
코드에서 groups는 1로 변하지 않음. 따라서 num_or_size_splits가 정수일 때 value차원을 axis을 따라 작은 텐서로 나눈다
```

```python
 featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]는
 featurmap = []
    for t1,t2 in zip(xNew, wNew):
        featurmap.append(conv(t1,t2))을 의미한다. zip(xNew, wNew)을 통해 t1,t2에 xNew와 wNew의 원소들이 하나씩 들어간다 ex)xNew=[1,2,3,4]이고 wNew=[5,6,7,8]이면 t1,t2 = 1,5 다음턴에는 2,6으로 차례대로 대입된다.
```

```python
mergeFeatureMap = tf.concat(axis = 3, values = featureMap) 3번째 axis를 더한다.
```

```python
out = tf.nn.bias_add(mergeFeatureMap, b) bias를 더해주는 코드
```

```python
return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name) tf.reshape을 통해 shape을 바꿔주고 activate function으로 relu를 사용한다.
```

## Class alexNet()

### __init__()

```python
def __init__(self, x, keepPro, classNum, skip, modelPath = "bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        #build CNN
        self.buildCNN()
```

### buildCNN()

```python
   def buildCNN(self):
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")
        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")
        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")
```

코드리뷰

```python

   def buildCNN(self):
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)
        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")
        
5개의 conv레이어와 3개 fc레이어 fc레이어 한뒤 dropout사용 2개의 dropout 사용
3번째  3 4번째 conv에서 padding="same"이며 maxpooling을 하지 않음
2번째 conv레이어 부터 groups=2로 설정
fcLayer는 다 True를 사용하여  return tf.nn.relu(out)을 사용

```



### loadModel()

```python
def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        #for layers in model
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            sess.run(tf.get_variable('w', trainable = False).assign(p))
```

## testModel

```python
import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes

parser = argparse.ArgumentParser(description='Classify some images.')
parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
parser.add_argument('-p', '--path', help='Specify a path [e.g. testModel]', default = 'testModel')
args = parser.parse_args(sys.argv[1:])

if args.mode == 'folder':
    #get testImage
    withPath = lambda f: '{}/{}'.format(args.path,f)
    testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    testImg = {args.path:url2img(args.path)}

# noinspection PyUnboundLocalVariable
if testImg.values():
    #some params
    dropoutPro = 1
    classNum = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            #img preprocess
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key,res))
            cv2.imshow("demo", img)
            cv2.waitKey(0)
```

코드리뷰

```python
if args.mode == 'folder':
    #get testImage
    withPath = lambda f: '{}/{}'.format(args.path,f)
    testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    testImg = {args.path:url2img(args.path)}
    
    
    "python3 testModel.py -m folder -p testModel"
    "python3 testModel.py -m url -p "
	으로 실행하게 만듬  url이미지 또는 디렉토리에있는 이미지
```

```python
if testImg.values():
    #some params
    dropoutPro = 1
    classNum = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)
    
    AlexNet은 256x256이미지를 227x227로 학습데이터를 늘려 오버피팅을 방지 마지막에 softmax로 분류해줌
    
      for key,img in testImg.items():
            #img preprocess
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean  이미지 사이즈 조정해줌
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]  softmax로 정한 class이름을 res로 설정

            font = cv2.FONT_HERSHEY_SIMPLEX		
            cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)		이미지 위에 class이름 씀
            print("{}: {}\n----".format(key,res))
            cv2.imshow("demo", img) 읽어들인 이미지 파일을 윈도우창에 보여준다
            cv2.waitKey(0) keyboard입력을 대기하는 함수로 0이면 key입력까지 무한대
```

# 이슈 

2개의 gpu를 사용하여 두 개로 나눠 학습하는 코드가 없음