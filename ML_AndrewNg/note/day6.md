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

### 6.6 高级优化

在上一个视频中，我们讨论了用梯度下降的方法最小化逻辑回归中代价函数J(θ)。在本次视频中，我会教你们一些高级优化算法和一些高级的优化概念，利用这些方法，我们就能够使通过梯度下降，进行逻辑回归的速度大大提高，而这也将使算法更加适合解决大型的机器学习问题。

另一种考虑梯度下降的思路是：我们需要写出代码来计算J(θ)和这些偏导数，然后把这些插入到梯度下降中，然后它就可以为我们最小化这个函数。 

如果我们能用这些方法来计算代价函数J(θ)和

偏导数项![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps2.jpg)两个项的话，那么这些算法就是为我们优化代价函数的不同方法，**共轭梯度法 BFGS(变尺度法)** 和**L-BFGS(限制变尺度法)** 就是其中一些更高级的优化算法。

它们可以自动尝试不同的学习速率α，并自动选择一个好的学习速率α，因此它甚至可以为每次迭代选择不同的学习速率，那么你就不需要自己选择。

我希望你们从这个幻灯片中学到的主要内容是：写一个函数，它能返回代价函数值、梯度值，因此要把这个应用到逻辑回归，或者甚至线性回归中，你也可以把这些优化算法用于线性回归，你需要做的就是输入合适的代码来计算这里的这些东西。

### 6.7 多类别分类：一对多

在本节视频中，我们将谈到如何使用逻辑回归 (**logistic regression**)来解决多类别分类问题，具体来说，我想通过一个叫做"一对多" (**one-vs-all**) 的分类算法。

对于一个多类分类问题，我们的数据集或许看起来像这样：

![image-20200508160047011](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508160047011.png)

我们现在已经知道如何进行二元分类，可以使用逻辑回归，对于直线或许你也知道，可以将数据集一分为二为正类和负类。用一对多的分类思想，我们可以将其用在多类分类问题上。

下面将介绍如何进行一对多的分类工作，有时这个方法也被称为"一对余"方法。

![image-20200508160502426](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508160502426.png)

现在我们有一个训练集，好比上图表示的有3个类别，我们用三角形表示 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps3.jpg)，方框表示![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps4.jpg)，叉叉表示 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps5.jpg)。我们下面要做的就是使用一个训练集，将其分成3个二元分类问题。

我们先从用三角形代表的类别1开始，实际上我们可以创建一个，新的"伪"训练集，类型2和类型3定为负类，类型1设定为正类，我们创建一个新的训练集，如下图所示的那样，我们要拟合出一个合适的分类器。

![image-20200508160529963](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508160529963.png)

最后我们得到一系列的模型简记为： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps6.jpg)其中：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps7.jpg) 

最后，在我们需要做预测时，我们将所有的分类机都运行一遍，然后对每一个输入变量，都选择最高可能性的输出变量。

我们要做的就是在我们三个分类器里面输入x，

然后我们选择一个让h^(i)(x)最大的i，即

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps8.jpg)。