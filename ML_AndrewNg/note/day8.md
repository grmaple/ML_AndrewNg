## 八、神经网络：表述(Neural Networks: Representation)

### 8.1 非线性假设

我们之前学的，无论是线性回归还是逻辑回归都有这样一个缺点，即：当特征太多时，计算的负荷会非常大。

之前我们已经看到过，使用非线性的多项式项，能够帮助我们建立更好的分类模型。

假设我们有非常多的特征，例如大于100个变量，我们希望用这100个特征来构建一个非线性的多项式模型，结果将是数量非常惊人的特征组合，即便我们只采用两两特征的组合![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps1.jpg)，我们也会有接近5000个组合而成的特征。这对于一般的逻辑回归来说需要计算的特征太多了。

普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。

### 8.2 神经元和大脑

神经网络是一种很古老的算法，它最初产生的目的是制造能模拟大脑的机器。

神经网络可能为我们打开一扇进入遥远的人工智能梦的窗户，但我在这节课中讲授神经网络的原因，主要是对于现代机器学习应用。它是最有效的技术方法。因此在接下来的一些课程中，我们将开始深入到神经网络的技术细节。

### 8.3 模型表示1

神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫激活单元，***\*activation unit\****）采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被成为权重（***\*weight\****）。

![image-20200523210633542](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523210633542.png)

我们设计出了类似于神经元的神经网络，效果如下：

![image-20200523210642120](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523210642120.png)

其中x1，x2，x3是输入单元（***\*input units\****），我们将原始数据输入给它们。a1，a2，a3是中间单元，它们负责将数据进行处理，然后呈递到下一层.最后是输出单元，它负责计算![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps2.jpg)。

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为输入层（***\*Input Layer\****），最后一层称为输出层（***\*Output Layer\****），中间一层成为隐藏层（***\*Hidden Layers\****）。我们为每一层都增加一个偏差单位（***\*bias unit\****）：

![image-20200523210731064](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523210731064.png)

上图所示的神经网络中![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps3.jpg)的尺寸为 3*4。输出为3，输入为4

对于上图所示的模型，激活单元和输出分别表达为：![image-20200523211051810](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211051810.png)

上面进行的讨论中只是将特征矩阵中的一行（一个训练实例）喂给了神经网络，我们需要将整个训练集都喂给我们的神经网络算法来学习模型。

我们可以知道：每一个a都是由上一层所有的x和每一个x所对应的θ决定的。（我们把这样从左到右的算法称为前向传播算法( ***\*FORWARD PROPAGATION\**** )）

把x，θ，a分别用矩阵表示，我们可以得到θ·X=a

### 8.4 模型表示2

( ***\*FORWARD PROPAGATION\**** ) 相对于使用循环来编码，利用向量化的方法会使得计算更为简便。以上面的神经网络为例，试着计算第二层的值：

![image-20200523211321900](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211321900.png)

![image-20200523211346768](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211346768.png)

如果我们要对整个训练集进行计算，我们需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即：

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps4.jpg)

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps5.jpg)

其实神经网络就像是***\*logistic regression\****，只不过我们把***\*logistic regression\****中的输入向量![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps6.jpg) 变成了中间层的![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps7.jpg), 即: 

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps8.jpg) 

我们可以把a0，a1，a2，a3看成更为高级的特征值，也就是x0，x2，x3，x4的进化体，并且它们是由x与θ决定的，因为是梯度下降的，所以a是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将x次方厉害，也能更好的预测新数据。 

这就是神经网络相比于逻辑回归和线性回归的优势。

### 8.5 特征和直观理解1

在神经网络中，原始特征只是输入层，在我们上面三层的神经网络例子中，第三层也就是输出层做出的预测利用的是第二层的特征，而非输入层中的原始特征，我们可以认为第二层中的特征是神经网络通过学习后自己得出的一系列用于预测输出变量的新特征。

神经网络中，单层神经元（无中间层）的计算可用来表示逻辑运算，比如逻辑与(***\*AND\****)、逻辑或(***\*OR\****)。

举例说明：逻辑与(***\*AND\****)；下图中左半部分是神经网络的设计与***\*output\****层表达式，右边上部分是***\*sigmod\****函数，下半部分是真值表。

我们可以用这样的一个神经网络表示***\*AND\**** 函数：

![image-20200523211723183](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211723183.png)

其中![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps10.jpg) 我们的输出函数![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps11.jpg)即为：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps12.jpg)

![image-20200523211734766](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211734766.png)

接下来再介绍一个***\*OR\****函数：

![image-20200523211805104](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211805104.png)

### 8.6 样本和直观理解II

二元逻辑运算符（***\*BINARY LOGICAL OPERATORS\****）当输入特征为布尔值（0或1）时，我们可以用一个单一的激活层可以作为二元逻辑运算符，为了表示不同的运算符，我们只需要选择不的权重即可。

![image-20200523211838453](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211838453.png)

我们可以利用神经元来组合成更为复杂的神经网络以实现更复杂的运算。例如我们要实现***\*XNOR（同或）\**** 功能（输入的两个值必须一样，均为1或均为0），即：

 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps13.jpg) 

首先构造一个能表达![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps14.jpg)部分的神经元：

![image-20200523211905803](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211905803.png)

然后将表示 ***\*AND\**** 的神经元和表示![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps15.jpg)的神经元以及表示 OR 的神经元进行组合：

![image-20200523211921171](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523211921171.png)

我们就得到了一个能实现XNOR 运算符功能的神经网络。

按这种方法我们可以逐渐构造出越来越复杂的函数，也能得到更加厉害的特征值。

这就是神经网络的厉害之处。

### 8.7 多类分类

当我们有不止两种分类时（也就是![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps17.jpg)），比如以下这种情况，该怎么办？如果我们要训练一个神经网络算法来识别路人、汽车、摩托车和卡车，在输出层我们应该有4个值。例如，第一个值为1或0用于预测是否是行人，第二个值用于判断是否为汽车。

输入向量![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps18.jpg)有三个维度，两个中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps19.jpg)，且![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml384\wps20.jpg)中仅有一个为1，表示当前类。下面是该神经网络的可能结构示例：

![image-20200523212036446](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523212036446.png)

![image-20200523212045063](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523212045063.png)

神经网络算法的输出结果为四种可能情形之一：

![image-20200523212053132](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200523212053132.png)