## 九、神经网络的学习(Neural Networks: Learning)

### 9.1 代价函数

首先引入一些便于稍后讨论的新标记方法：

假设神经网络的训练样本有m个，每个包含一组输入x和一组输出信号y，L表示神经网络层数，Sl表示每层的神经元个数，SL代表最后一层中处理单元的个数。

将神经网络的分类定义为两种情况：二类分类和多类分类，

二类分类：SL=0，y=0or1表示哪一类；

K类分类：SL=k，yi=1表示分到第i类（k>2）

![image-20200606152315384](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606152315384.png)

我们回顾逻辑回归问题中我们的代价函数为：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps1.png)

在逻辑回归中，我们只有一个输出变量，又称标量（***\*scalar\****），也只有一个因变量y，但是在神经网络中，我们可以有很多输出变量，我们的hθ（x）是一个维度为K的向量，并且我们训练集中的因变量也是同样维度的一个向量，因此我们的代价函数会比逻辑回归更加复杂一些，为：

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps3.jpg) ，![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps4.jpg)

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps5.png)

这个看起来复杂很多的代价函数背后的思想还是一样的，我们希望通过代价函数来观察算法预测的结果与真实情况的误差有多大，唯一不同的是，对于每一行特征，我们都会给出K个预测，基本上我们可以利用循环，对每一行特征都预测K个不同结果，然后在利用循环在K个预测中选择可能性最高的一个，将其与y中的实际数据进行比较。

正则化的那一项只是排除了每一层θ0后，每一层的θ矩阵的和。最里层的循环j循环所有的行（又sl+1层的激活单元数决定），循环i则循环所有的列，由该层（sl层）的激活单元数所决定。

即：hθ(x)与真实值之间的距离为每个样本-每个类输出的加和，对参数进行***regularization***的***bias***项处理所有参数的平方和。

### 9.2 反向传播算法

之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法，我们从第一层开始正向一层一层进行计算，直到最后一层的hθ(x)。

现在，为了计算代价函数的偏导数![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps6.jpg)，我们需要采用一种反向传播算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。 以一个例子来说明反向传播算法。

假设我们的训练集只有一个实例![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps7.jpg)，我们的神经网络是一个四层的神经网络，其中K=4，SL=4，L=4

前向传播算法：

![image-20200606153013923](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606153013923.png)

![image-20200606153018030](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606153018030.png)

我们从最后一层的误差开始计算，误差是激活单元的预测（![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps8.png)）与实际值（![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps9.jpg)）之间的误差，

我们用δ来表示误差，则：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps10.jpg)

我们利用这个误差值来计算前一层的误差：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps11.jpg) 其中 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps12.jpg)是S形函数的导数，![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps13.jpg)。而![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps14.jpg)则是权重导致的误差的和。下一步是继续计算第二层的误差： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps15.jpg) 

因为第一层是输入变量，不存在误差。我们有了所有的误差的表达式后，便可以计算代价函数的偏导数了，假设λ=0，即我们不做任何正则化处理时有： ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps16.jpg)

重要的是清楚地知道上面式子中上下标的含义：

l代表目前所计算的是第几层。

j代表目前计算层中的激活单元的下标，也将是下一层的第j个输入变量的下标。

i代表下一层中误差单元的下标，是受到权重矩阵中第i行影响的下一层中的误差单元的下标。

如果我们考虑正则化处理，并且我们的训练集是一个特征矩阵而非向量。在上面的特殊情况中，我们需要计算每一层的误差单元来计算代价函数的偏导数。在更为一般的情况中，我们同样需要计算每一层的误差单元，但是我们需要为整个训练集计算误差单元，此时的误差单元也是一个矩阵，我们用![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps17.jpg)来表示这个误差矩阵。第l层的第i个激活单元受到第j个参数影响而导致的误差。

我们的算法表示为：

![image-20200606153437858](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606153437858.png)

即首先用正向传播方法计算出每一层的激活单元，利用训练集的结果与神经网络预测的结果求出最后一层的误差，然后利用该误差运用反向传播法计算出直至第二层的所有误差。

在求出了![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps18.jpg)之后，我们便可以计算代价函数的偏导数了，计算方法如下：

<img src="file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps19.jpg" alt="img" style="zoom:67%;" /> ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps20.jpg)

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps21.jpg) ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps22.jpg)

### 9.3 反向传播算法的直观理解

在这段视频中，我想更加深入地讨论一下反向传播算法的这些复杂的步骤，并且希望给你一个更加全面直观的感受，理解这些步骤究竟是在做什么，也希望通过这段视频，你能理解，它至少还是一个合理的算法。但可能你即使看了这段视频，你还是觉得反向传播依然很复杂，依然像一个黑箱，太多复杂的步骤，依然感到有点神奇，这也是没关系的。即使是我接触反向传播这么多年了，有时候仍然觉得这是一个难以理解的算法，但还是希望这段视频能有些许帮助，为了更好地理解反向传播算法，我们再来仔细研究一下前向传播的原理：

前向传播算法：

<img src="C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155056726.png" alt="image-20200606155056726" style="zoom: 67%;" />

![image-20200606155101054](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155101054.png)

反向传播算法做的是：

![image-20200606155123480](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155123480.png)

![image-20200606155128332](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155128332.png)

### 9.4 实现注意：展开参数

在上一段视频中，我们谈到了怎样使用反向传播算法计算代价函数的导数。在这段视频中，我想快速地向你介绍一个细节的实现过程，怎样把你的参数从矩阵展开成向量，以便我们在高级最优化步骤中的使用需要。

![image-20200606155152726](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155152726.png)

![image-20200606155155668](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155155668.png)

![image-20200606155158810](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155158810.png)

### 9.5 梯度检验

当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。

为了避免这样的问题，我们采取一种叫做梯度的数值检验（***\*Numerical Gradient Checking\****）方法。这种方法的思想是通过估计梯度值来检验我们计算的导数值是否真的是我们要求的。

对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的平均值用以估计梯度。即对于某个特定的θ，我们计算出在θ-ε处和θ+ε 的代价值（ε是一个非常小的值，通常选取 0.001），然后求两个代价的平均，用以估计在θ处的代价值。

![image-20200606155331364](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155331364.png)

当θ是一个向量时，我们则需要对偏导数进行检验。因为代价函数的偏导数检验只针对一个参数的改变进行检验，下面是一个只针对θ1进行检验的示例：

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps23.jpg)

最后我们还需要对通过反向传播方法计算出的偏导数进行检验。

根据上面的算法，计算出的偏导数存储在矩阵

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps24.jpg) 中。检验时，我们要将该矩阵展开成为向量，同时我们也将 θ 矩阵展开为向量，我们针对每一个θ 都计算一个近似的梯度值，将这些值存储于一个近似梯度矩阵中，最终将得出的这个矩阵同 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps25.jpg) 进行比较。

![image-20200606155434209](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200606155434209.png)

### 9.6 随机初始化

任何优化算法都需要一些初始的参数。到目前为止我们都是初始所有参数为0，这样的初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。如果我们令所有的初始参数都为0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我们初始所有的参数都为一个非0的数，结果也是一样的。

我们通常初始参数为正负ε之间的随机值，假设我们要随机初始一个尺寸为10×11的参数矩阵，代码如下：

Theta1 = rand(10, 11) * (2*eps) – eps

### 9.7 综合起来

小结一下使用神经网络时的步骤：

网络结构：第一件要做的事是选择网络结构，即决定选择多少层以及决定每层分别有多少个单元。

第一层的单元数即我们训练集的特征数量。

最后一层的单元数是我们训练集的结果的类的数量。

如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。

我们真正要决定的是隐藏层的层数和每个中间层的单元数。

训练神经网络：

1. 参数的随机初始化
2. 利用正向传播方法计算所有的![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps27.jpg)
3. 编写计算代价函数 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml12916\wps28.jpg) 的代码
4. 利用反向传播方法计算所有偏导数
5. 利用数值检验方法检验这些偏导数
6. 使用优化算法来最小化代价函数