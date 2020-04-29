## 六、逻辑回归

### 6.1 分类问题

在分类问题中，你要预测的变量y是离散的值，我们将学习一种叫做**逻辑回归** 的算法，这是目前最流行使用最广泛的一种学习算法。

在分类问题中，我们尝试预测的是结果是否属于某一个类（例如正确或错误）。分类问题的例子有：判断一封电子邮件是否是垃圾邮件；判断一次金融交易是否是欺诈；之前我们也谈到了肿瘤分类问题的例子，区别一个肿瘤是恶性的还是良性的。

我们从二元的分类问题开始讨论。

我们将**因变量**可能属于的两个类分别称为**负向类**和**正向类**，则因变量y∈0,1，其中0表示负向类，1表示正向类。

![image-20200429204305689](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429204305689.png)

逻辑回归算法是分类算法，我们将它作为分类算法使用。有时候可能因为这个算法的名字中出现了“回归”使你感到困惑，但逻辑回归算法实际上是一种分类算法。

### 6.2 假说表示

在这段视频中，我要给你展示假设函数的表达式，也就是说，在分类问题中，要用什么样的函数来表示我们的假设。此前我们说过，希望我们的分类器的输出值在0和1之间，因此，我们希望想出一个满足某个性质的假设函数，这个性质是它的预测值要在0和1之间。

逻辑回归模型的假设是： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps2.jpg)

其中：X代表特征向量，g代表**逻辑函数**是一个常用的逻辑函数为S形函数，公式为： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps3.jpg)

该函数的图像为：

![image-20200429205352729](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429205352729.png)

合起来，我们得到逻辑回归模型的假设：

对模型的理解： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps4.jpg)。

h(x)的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性，

即![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps5.jpg) 

### 6.3 判定边界

现在讲下**决策边界**的概念。这个概念能更好地帮助我们理解逻辑回归的假设函数在计算什么。

![image-20200429210722745](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429210722745.png)

并且参数θ是向量[-3 1 1]。 则当-3+x1+x2≥0，即x1+x2≥3时，模型将预测y=1。

我们可以绘制直线x1+x2=3，这条线便是我们模型的分界线，将预测为1的区域和预测为 0的区域分隔开。

<img src="C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429210913653.png" alt="image-20200429210913653" style="zoom:80%;" />

假使我们的数据呈现这样的分布情况，怎样的模型才能适合呢？

![image-20200429210933172](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429210933172.png)

因为需要用曲线才能分隔y=0的区域和y=1的区域，我们需要二次方特征：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps6.jpg)θ是[-1 0 0 1 1]，则我们得到的判定边界恰好是圆点在原点且半径为1的圆形。

### 6.4 代价函数

这段视频中，我们要介绍如何拟合逻辑回归模型的参数θ。具体来说，我要定义用来拟合参数的优化目标或者叫代价函数，这便是监督学习问题中的逻辑回归模型的拟合问题。

![image-20200429212232787](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429212232787.png)

我们重新定义逻辑回归的代价函数为：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps7.jpg)

，其中

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps8.jpg) 

h(x)与 cost(h(x),y)之间的关系如下图所示：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps11.png) 

### 6.5 简化的成本函数和梯度下降

![image-20200429214837728](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429214837728.png)

这个式子可以合并成：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps12.jpg) 

即，逻辑回归的代价函数：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml3016\wps17.jpg) 

最小化代价函数的方法，是使用**梯度下降法**

![image-20200429215119757](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429215119757.png)

现在，如果你把这个更新规则和我们之前用在线性回归上的进行比较的话，你会惊讶地发现，这个式子正是我们用来做线性回归梯度下降的。

那么，线性回归和逻辑回归是同一个算法吗？要回答这个问题，我们要观察逻辑回归看看发生了哪些变化。实际上，假设的定义发生了变化。

推导过程：

<img src="C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200429215229539.png" alt="image-20200429215229539" style="zoom:150%;" />

另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的。