# Cifar-10 Classification 코드 리뷰

```python
import tensorflow as tf 
import numpy as np 
 
  
 from tensorflow.keras.datasets.cifar10 import load_data
#Cifar-10 데이터셋 import
 
 
 # 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의한다. 
 def next_batch(num, data, labels): 
   idx = np.arange(0 , len(data)) 
   np.random.shuffle(idx) 
   idx = idx[:num] 
   data_shuffle = [data[ i] for i in idx] 
   labels_shuffle = [labels[ i] for i in idx] 
   return np.asarray(data_shuffle), np.asarray(labels_shuffle) 
# 다음 배치를 읽어오기 위해 랜덤으로 샘플과 레이블을 돌려 리턴해줌


 def build_CNN_classifier(x): 
   
   x_image = x 
    # 입력 이미지 
 
  
   W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2)) 
	#필터를 의미, 크기5x5 필터의 깊이 3 필터 개수 64
   b_conv1 = tf.Variable(tf.constant(0.1, shape=[64])) 
   h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1) 
    #첫번째 conv2d에 bias를 더하고 activate 함수로 relu 사용
 
  
   h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') 
  # 첫번째 Pooling layer tf.nn.max_pool을 사용하여 maxpooling을 함 이때 필터는 strides가 2이므로 두칸씩이동  
 
   
   W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2)) 
   b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) 
   h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2) 
	# 두번째 convolutional layer - 32개의feature를 64개의 feature들로 맵핑한다. 첫번 째 convolution레이아웃과 똑같음
 
 
   
   h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') 
  # 두번째 pooling layer 첫번째와 똑같이 max풀링한다.
 
  
   W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2)) 
   b_conv3 = tf.Variable(tf.constant(0.1, shape=[128])) 
   h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3) 
     # 세번째 convolutional layer 64개의 feature에서 128개의 feature로 맵핑한다. 이 단계부터 max풀링을 하지 않음
 
 
  
   W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2)) 
   b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))  
   h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4) 
     # 네번째 convolutional layer 1~3번째 convolutional layer과 똑같이 동작
 
 
   
   W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2)) 
   b_conv5 = tf.Variable(tf.constant(0.1, shape=[128])) 
   h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5) 
    # 다섯번째 convolutional layer 1~4번째 convolutional layer과 똑같이 동작 maxpooling하지 않음 여기서 8x8x128 feature map이 생성
 
 
   

   W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2)) 
   b_fc1 = tf.Variable(tf.constant(0.1, shape=[384])) 
    #384개의 feature로 맵핑한다 
 
   h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128]) 
   #tf.reshape을 통해 h_conv5의 shape을 바꿔줌
   h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1) 
   #h_conv5_flat과 W_fc1을 곱하고 bias를 더해준 값을 relu함수의 넣음. FC_layer1
 

   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#tf.nn.dropout을 통해 오버피팅을 방지 dropout이란 훈련data에서만 좋은 성과를 내는 Overfitting을 막기위해 해주는 것으로 몇개의 노드들을 랜덤하게 죽이고 남은 노드들을 통해서만 훈련한다. 
 
 
  
   W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2)) 
   b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) 
   logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2 
   y_pred = tf.nn.softmax(logits) 
   return y_pred, logits
 # FC_layer2 3 softmax를 통해 10개의 클래스로 예측. 
 
 
 
 x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3]) 
 y = tf.placeholder(tf.float32, shape=[None, 10]) 
 keep_prob = tf.placeholder(tf.float32) 
 #값들을 던저주기 위해 x,y,keep_prob을 placeholder로 정의
 
 

 (x_train, y_train), (x_test, y_test) = load_data() 
  # CIFAR-10 데이터 불러오기    
 y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1) 
 y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1) 
#  One-hot인코딩을 통해 스칼라형태의 레이블을 레이블에 맞으면 1로 만듬
 
 y_pred, logits = build_CNN_classifier(x) 
 
 

 loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) 
#학습하기위한 loss를 계산 
 train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss) 
  #RMSPropOptimizer를 이용해서 비용 함수를 최소화한다고 한다.
 
 
 # 정확도를 계산하는 연산을 추가합니다. 
 correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)) 
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
 
 

 with tf.Session() as sess:  
   sess.run(tf.global_variables_initializer()) 
	#변수들 초기화
    
   # 10000 Step만큼 최적화를 수행합니다. 
   for i in range(10000): 
     batch = next_batch(128, x_train, y_train_one_hot.eval()) 
    #한번의 학습할 양 batch
 

    
     if i % 100 == 0: 
       train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}) 
       loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}) 
        # 100 스탭을 주기로 정확도와 loss를 출력합니다. 
 
 
       print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print)) 
     # 20% 확률의 Dropout을 이용해서 학습을 진행합니다. 
     sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8}) 
   
   test_accuracy = 0.0   
   for i in range(10): 
     test_batch = next_batch(1000, x_test, y_test_one_hot.eval()) 
     test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0}) 
    #feed_dict로 placeholder값 전달
   test_accuracy = test_accuracy / 10; 
   print("테스트 데이터 정확도: %f" % test_accuracy) 

   # 학습이 끝나면 테스트 데이터에 대한 정확도를 출력한다.

```



