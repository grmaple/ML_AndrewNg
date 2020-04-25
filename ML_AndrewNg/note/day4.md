## 四、多变量线性回归

### 4.1多维特征

目前为止，我们探讨了单变量/特征的回归模型，现在我们对房价模型增加更多的特征，例如房间数楼层等，构成一个含有多个变量的模型，模型中的特征为(x1,x2,...,xn)

![image-20200425161436381](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425161436381.png)

增添更多特征后，我们引入一系列新的注释：

n代表特征的数量

x^(i)代表第i个训练实例，是特征矩阵中的第i行，是一个**向量**

比方说，上图的

![image-20200425161601505](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425161601505.png)

xj^(i)代表特征矩阵中第i行的第j个特征，也就是第i个训练实例的第j个特征

如上图的x2^(2)=3,x3^(2)=2

支持多变量的假设h表示为：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps1.jpg)

这个公式中有n+1个参数和n个变量，为了使得公式能够简化一些，引入x0=1，则公式转化为：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps2.jpg)

特征矩阵X的维度是m*(n+1)因此公式可以简化为:![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps3.jpg)



### 4.2多元梯度下降法

与单变量线性回归类似，在多变量线性回归中，我们也构建一个代价函数，则这个代价函数是所有建模误差的平方和，即：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps4.jpg)

其中：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps5.jpg)

们的目标和单变量线性回归问题中一样，是要找出使得代价函数最小的一系列参数。 多变量线性回归的批量梯度下降算法为：

![image-20200425162051080](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162051080.png)

我们开始随机选择一系列的参数值，计算所有的预测结果后，再给所有的参数一个新的值，如此循环直到收敛。

### 4.3多元梯度下降法实践-特征缩放

在我们面对多维特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。

以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。

![image-20200425162142637](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162142637.png)

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。

![image-20200425162154048](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162154048.png)

最简单的方法是令：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps6.jpg)，其中μn是平均值,sn是标准差（范围）。

### 4.4多元梯度下降法实践-学习率

梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，我们不能提前预知，我们可以绘制迭代次数和代价函数的图表来观测算法在何时趋于收敛。

![image-20200425162334792](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162334792.png)

也有一些自动测试是否收敛的方法，例如将代价函数的变化值与某个阀值（例如0.001）进行比较，但通常看上面这样的图表更好。

梯度下降算法的每次迭代受到学习率的影响，如果学习率α过小，则达到收敛所需的迭代次数会非常高；如果学习率α过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。

通常可以考虑尝试些学习率：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps11.jpg) 

### 4.5特征和多项式回归

如房价预测问题，

<img src="C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162440804.png" alt="image-20200425162440804" style="zoom:50%;" />

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps12.jpg) 

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps13.jpg)则：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps14.jpg)。 

线性回归并不适用于所有数据，有时我们需要曲线来适应我们的数据，比如一个二次方模型：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps15.jpg) 

或者三次方模型： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps16.jpg) 

![image-20200425162628653](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162628653.png)

通常我们需要先观察数据然后再决定准备尝试怎样的模型。

外，我们可以令：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps17.jpg)，从而将模型转化为线性回归模型。

根据函数图形特性，我们还可以使：

![image-20200425162729338](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162729338.png)

注：如果我们采用多项式回归模型，在运行梯度下降算法前，特征缩放非常有必要。

### 4.6正规方程

到目前为止，我们都在使用梯度下降算法，但是对于某些线性回归问题，正规方程方法是更好的解决方案。如：

![image-20200425162757647](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162757647.png)

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps19.jpg) 。  假设我们的训练集特征矩阵为X（包含了x0=1）并且我们的训练集结果为向量y，则利用正规方程解出向量 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml11916\wps20.jpg)

![image-20200425162936285](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425162936285.png)

梯度下降与正规方程的比较：

![image-20200425163034533](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425163034533.png)

总结一下，只要特征变量的数目并不大，标准方程是一个很好的计算参数θ的替代方法。具体地说，只要特征变量数量小于一万，我通常使用标准方程法，而不使用梯度下降法。

正规方程可以不用特征缩放。

![image-20200425163311298](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200425163311298.png)

