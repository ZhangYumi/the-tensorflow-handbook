# MNIST手写数字数据集介绍
---
MNIST数据集是一个有名的手写数字数据集，在深度学习领域，手写数字识别是一个很经典的学习例子。MNIST数据集由四部分组成，本文我们将对MNIST数据集作一个初步了解。

* 首先，我们下载MNIST数据集。

![](/assets/TIM截图20180430203925.png)

导入相关python模块，运行将会自动下载MNIST数据集至项目文件夹。比如我个人的是在目录“E:\Tensorflow_Study\”目录下。

![](/assets/TIM截图20180430204259.png)

* 下载好MNIST数据集后，我们可以打印输出训练集、测试集、验证集的图像及标签信息。

![](/assets/TIM截图20180430210229.png)

![](/assets/TIM截图20180430210513.png)

从上面运行结果可以看出，训练集图像有55000张，每张大小为784=28*28；训练集标签为55000个，每个标签为长度为10的一维数组；测试集10000张图片，验证集5000张图片。
* 接下来，我们绘制其中一张图片，通过可视化直观地看一下手写数字的图片。

![](/assets/捕获.PNG)

我们用训练集的第一张图片作为例子可视化输出，输出原图像及对应灰度图像，对应数字为7。下一篇文章中，我们将用Tensorflow构建一个神经网络，针对MNIST数据集进行手写数字识别训练。

本文可视化输出手写数字图片完整代码如下：

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets(r"E:\\TensorFlow_Study\\MNIST_data\\", one_hot=True)

img = mnist.train.images[0]
label = mnist.train.labels[0]

plt.figure()

plt.subplot(1,2,1)
plt.imshow(img.reshape(28,28))
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img.reshape(28,28),cmap='gray')
plt.axis('off')

plt.show()
```

