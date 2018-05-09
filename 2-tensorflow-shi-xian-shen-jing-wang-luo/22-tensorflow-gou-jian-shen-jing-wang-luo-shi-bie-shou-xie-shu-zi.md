# Tensorflow构建神经网络识别手写数字
---
在前面的文章中，我们针对MNIST数据集以及神经网络进行了简单的介绍，本文在此基础上，将针对MNIST数据集，利用Tensorflow构建一个神经网络，训练网络对MNIST手写数字进行识别。

* ###首先，我们导入相关python模块以及MNIST数据集。

![](/assets/TIM截图20180501004018.png)

“one_hot=True”表示用0,1元素组成的长度为10的向量表示标签数字0~9。比如数字3表示成[0,0,0,1,0,0,0,0,0,0],数字3为向量中元素1的索引。

* ###定义神经网络的输入输出以及隐藏层的参数。
![](/assets/微信截图_20180509144728.png)

其中，**tf.palceholder()**定义一个占位符，从内存中读入数据（关于tensorflow读取数据我们会在相关章节作专门的详细讲解）。**tf.nn.softmax()**的输入是一个向量，输出是一个归一化后的向量，用于计算向量中各个元素占元素总和的权重。例如，向量 A = $$[a1, a2, a3]$$,那么**tf.nn.softmax(A)**的输出为$$[a1/∑ai, a2/∑ai,a3/∑ai]$$(其中，$$∑ai = a1 + a2 +a3$$)。

1. 神经网络的输入x为[None,784],第一个为输入的图片数量，每张大小为784。由于暂不确定张数，用None占位。比如我们输入100张图片，那么x为100*784矩阵。

2. 神经网络的输入y\_为[None,10],第一个为输入的数字标签数量，上面我们说过，每个数字标签是一个长度为10的由0,1组成的数组。所以若我们输入100个数字标签，那么y\_为100*10矩阵。输入的图片（images）与标签（labels）数量相等，一一对应。

3. 隐藏层参数w为[784,10]矩阵，784行，10列。这儿的10可以理解是标签数字是一个十分类问题。b为长度为10的向量。w和b初始值我们这儿用0填充，调用**tf.zeros()**函数。
4. 神经网络的输出层y为 $$y = x*w + b$$,然后我们将y用**tf.nn.softmax()**映射到（0,1）范围内。用代码表示，输出层 **y = tf.nn.softmax(tf.matmul(x,w)+b)**。


* ###定义损失函数，构建神经网络。
![](/assets/微信截图_20180509202308.png)

损失函数 $$loss = ∑(y-y\_)²$$,用代码实现为**loss = tf.reduce_sum(tf.square(y-y__))**。如果神经网络对图片数字识别准确，那么y中权值最大的索引号与标签数字的对应的向量中1的索引号一致。即就是说，loss函数理论上取得一个比较小的值（权值小的值减去0，权值最大的值减去1,然后差的平方求和）。

**train_step =  tf.train.GradientDescentOptimizer(0.001).minimize(loss)**是指以0.001为学习率进行梯度下降，已达到loss局部最小化值。tensorflow的梯度下降函数会自动进行反向传播更新w和b的值，这一点无需我们自己去实现。

* ###创建会话，训练神经网络，输入验证集，计算准确率。

![](/assets/微信截图_20180510011041.png)


1. 通过代码**init = tf.global_variables_initializer()**初始化所有变量，并调用**sess.run(init)**计算这些变量，这在前面关于会话的章节中有所概述，此处我们就不多说了。

2. **mnist.train.next_batch(N)**:该函数从训练集中返回一个batch列表，包含两个batch：一个图片batch，包含N张图片；一个标签batch，包含N个数字标签。也就是说，将N张图片作为一个数据块，返回给batch\_xtrain；将N个标签向量作为一个数据块，返回给batch\_ytrain。所以batch\_xtrain为N\*784矩阵，batch\_ytrain为N\*10矩阵。

3. **sess.run(train_step,feed_dict={x:batch_xtrain,y_:batch_ytrain})**:该行代码表示将batch_xtrain,batch_ytrain作为输入数据训练神经网络。我们用for循环迭代训练2000次，每200次打印输出一次损失函数loss的值，观察loss的变化趋势。可以看到，我们的loss从初试的89降到了15（此处可以思考为什么loss不趋近于0）。

    ![](/assets/微信截图_20180510013756.png)

4. **tf.equal(A,B)**：对比两个矩阵或向量A和B的对应元素，元素相等就返回True，元素不等就返回False。比如A = [1,2,3,4], B = [1,2,3,5]。那么**tf.equal(A,B)**将返回[True,True,True,False]。

   **tf.argmax(vector,1)**：返回vector中的最大值索引号。若vector是一个向量，就返回最大值索引；若vector是一个矩阵，就返回每一个维度对应矩阵行的最大值索引。前面我们说过，当识别准确时，y与y\_的最大值索引号理论上相等，索引号相等则**tf.equal(tf.argmax(y,1),tf.argmax(y_,1))**返回True组成的数组。也就是说**correct\_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))**返回的数组correct\_prediction中，True的个数越多，识别准确度越高。
   
   **tf.cast(A,dtype)**:该函数是将矩阵A的元素进行数据类型转换。此处我们将correct\_prediction中的元素由bool型转换成float型。True变成1.0，False变成0。转换成数值型，便于我们计算精确度。
   
   **tf.reduce_mean(A)**:该函数求A中元素的平均值。
  
5. 上面我们介绍了怎么计算预测的准确度。接下来，我们用for循环迭代10个batch,每个batch为100个从测试集中随机抽取的测试样本。方法与训练集batch一样，此处不再赘述。然后我们用训练好的神经网络来识别每个batch中的数字图片，并计算预测准确度。可以看出，我们训练的神经网络在大部分测试集batch上都达到了90%以上的识别率。

![](/assets/微信截图_20180510021011.png)

* ###总结：
本文利用tensorflow实现了最简单的神经网络，实现手写数字识别。当然，本例没有做太多神经网络优化，比如损失函数loss过于粗糙，训练迭代次数等都会影响到神经网络的预测效果。在后续的学习中，我们会讲解一些神经网络的优化方法。

* ###附本例完整代码：

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r"E:\\TensorFlow_Study\\MNIST_data\\", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None,10])

loss = tf.reduce_sum(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        batch_xtrain,batch_ytrain=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xtrain,y_:batch_ytrain})
        if i%200==0:
            print("step_%d loss: %f"%(i,sess.run(loss,feed_dict={x:batch_xtrain,y_:batch_ytrain})))
            
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
    for i in range(10):
        batch_xtest, batch_ytest = mnist.test.next_batch(100)
        print("batch_%d accuracy: %.2f"%(i,sess.run(accuracy, feed_dict={x: batch_xtest, y_: batch_ytest})))

```

