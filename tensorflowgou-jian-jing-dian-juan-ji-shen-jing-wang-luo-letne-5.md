# Tensorflow构建经典卷积神经网络LeNet-5
---
前面我们对卷积神经网络（CNN）以及LeNet-5模型作了一个简单的介绍。本文我们将用Tensorflow构建一个LeNet-5模型，实现手写数字的识别。
### 1.代码整体结构
我们首先从代码的整体封装去理解卷积神经网络以及LeNet-5的结构。回忆一下，一个LeNet-5模型的前向传播过程为：输入层 → 卷积层1 → 池化层1 → 卷积层2 → 池化层2 → 全连接层1 → 全连接层2 → 输出层。神经网络的输入层我们需要初始化权重W和偏移量b，我们将这两个初始化函数进行封装为*get\_weight()*和*get\_bias()*,同时，卷积层与池化层我们这里自定义两个子函数：卷积计算函数_conv2d()_和最大池化计算函数_max_pooling()_。整体代码封装如下图：

![](/assets/TIM截图20180523152147.png)

### 2.参数定义
我们首先定义LeNet-5模型中各层输入输出的size参数以及depth参数。如图，输入层的输入为28×28的灰度图像，因此输入层输入为28×28×1的矩阵，IMAGE_SIZE为28，NUM\_CHANNELS为1。第一层卷积层，我们用5×5×32的过滤器对输入图像进行卷积，CONV1_SIZE为5，CONV1_KERNEL_NUM为32。卷积时采用0填充，因此输出图像为28×28×32。接下来，我们采用2×2的池化矩阵，以步长为2对上面的输出图像进行滑动最大池化，得到14×14×32。到目前，我们已经完成了第一层卷积和第一层最大池化。

接下来，我们将第一层池化的输出结果作为输入进行第二层卷积，此时输入为14×14×32，第二层卷积层过滤器为5×5×64，因此，CONV2_SIZE为5，CONV2_KERNEL_NUM为64，然后同样最大池化，得到输出为7×7×64。最后是全连接层，第一层全连接层的输入为7×7×64。设置全连接层权重参数向量长度为1024，最终输出向量是长度为10的0、1组成的向量。

![](/assets/TIM截图20180523154109.png)

### 3.初始化权重W

![](/assets/TIM截图20180523160520.png)

我们用tf.truncated_normal()函数进行初始化权重W。tf.truncated_normal()返回一个随机数矩阵，这些随机数服从一个截断的正态分布。tf.truncated_normal()的定义为：

```
tf.truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)              
```
shape是返回的随机数矩阵的形状，mean是均值，stddev是标准差，dtype是数据类型，seed是随机种子，name是返回矩阵的名称。这里我们将shape作为初始化参数，标准差stddev设置为0.1,标准差设置过大导致权重参数分布范围过大，更新不稳定，容易导致梯度爆炸。

### 4.初始化偏移量b

![](/assets/TIM截图20180523162738.png)

get_bias()我们定义两个参数，矩阵形状shape与参数初始化填充值value，默认为0.1。比如get_bias([3,2])将返回一个tensorflow变量，会话中运行该变量得到矩阵：

```
[[0.1,0.1],
 [0.1,0.1],
 [0.1,0.1]]
```

### 5.自定义卷积计算函数
我们当然可以在前向传播过程中调用tf.nn.conv2d()函数，但是更好的做法是对它进行有针对性的封装。

![](/assets/TIM截图20180523163542.png)

tf.nn.conv2d()函数的定义为：

```
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```
其中，input为输入矩阵，filter为过滤器矩阵，strides为滑动窗口步长的列表，其中第1个和第4个参数必须为1。假设我们设stride为步长，则 strides=[1,stride,stride,1]。padding为填充参数，前面我们介绍过，有"SAME"和"VALID"两种选择，前者代表用0填充，后者代表不填充，主要用到也就这些参数。

### 6.自定义最大池化函数

![](/assets/TIM截图20180523164747.png)

tf.nn.max_pool()函数的定义为：

```
tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```
其中，ksize的参数形式为[batch,height,width,channels]，batch与channels通常设置为1，height与width分别为池化矩阵的高和宽，其他与卷积类似，不做赘述。

### 7.前向传播函数
这一部分我们已经在前面讲述了很多，在参数定义部分我们一步一步计算了前向传播整个流程，因此这一部分我就不做过多讲述，在代码注释中我也一步一步进行了讲解，由于PC截长图比较麻烦，因此此处我并将前向传播函数的代码直接贴在下方，供大家参考：


```
def forward(x,train):
    """
    LetNe-5经典卷积神经网络前向传播
    :param x: 输入矩阵
    :param train: 是否进行dropout防过拟合优化
    :return y: 输出结果
    """
    #*************第一层卷积层**************#
    # 输入为x = 28x28x1,滤波器filter为5x5x32,卷积后输出为28x28x32
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM])    #patch 5x5, input size 1, output size 32
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))  #output size 28x28x32

    #*************第一层池化层***************#
    # 输入为28x28x32，输出为14x14x32
    pool1 = max_pool_2x2(relu1) #output size 14x14x32

    #*************第二层卷积层***************#
    # 输入为上面池化层输出14x14x32，滤波器filter为5x5x64，卷积后输出为14x14x64
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM]) #patch 5x5, input size 32, output size 64
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))  #output size 14x14x64

    #*************第二层池化层***************#
    # 输入为14x14x64，输出为7x7x64
    pool2 = max_pool_2x2(relu2)  #output size 7x7x64
    pool2_reshape = tf.reshape(pool2,[-1,7*7*64])    #矩阵变换

    #*************第一层全连接层**************#
    fc1_w = get_weight([7*7*64,FC_SIZE])
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(pool2_reshape,fc1_w)+fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1,0.5)    #按1-0.5的概率置0，其余的未置0的乘以1/0.5,保证总体期望值不变，防止过拟合。

    #*************第二层全连接层***************#
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE])
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.nn.softmax(tf.matmul(fc1, fc2_w) + fc2_b)

    return y

```

