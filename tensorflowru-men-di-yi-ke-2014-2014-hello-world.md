# Tensorflow入门第一课————Hello,world！
---
对于每一个计算机专业的学生或者IT行业工作者来说，接触一门新的语言或者技术，第一次使用该语言或技术总是习惯打印输出一个“Hello，world！”，这几乎成了一种约定。那么，本文就用Tensorflow输出一个“Hello，world！”，作为我们的Tensorflow入门第一课，以验证我们的Tensorflow环境是否安装配置正确，是否可以正常使用。

* 首先，我们导入tensorflow模块：

![](/assets/TIM图片20180430115942.png)

* 然后，我们定义一个Tensorflow常量，赋值为“Hello，world！”（关于Tensorflow的常量、变量等定义方式下文我将进行详细讲解），并打印输出该常量以及类型。

![](/assets/TIM截图20180430120547.png)

在这里我们发现一个问题，我们用constant定义一个常量a,并赋值“Hello，world！”，但打印输出a,得到的是Tensor("Const_1:0",shape=(),dtype=string),而a的type输出是一个Tensor。这里涉及到Tensorflow的计算图的概念，我们将在后续讲到。这里我们只需要知道，目前的常量a我们只定义了它的类型以及计算方式（比如赋值），但并未真正计算。为了输出a保存的内容，我们需要在会话（session）中进行。

* 接下来，我们定义一个会话，打印输出a的内容。

![](/assets/TIM截图20180430121815.png)

关于会话，我们也在后文作详细讲解。至此，我们就完成了我们的第一个Tensorflow程序————Hello，world！

* 附：本例完整代码如下：

```
import tensorflow as tf

a = tf.constant("Hello,world!")
sess = tf.Session()
print(sess.run(a))
sess.close()
```



