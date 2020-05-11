## 七、正则化(Regularization)

### 7.1 过拟合的问题

到现在为止，我们已经学习了几种不同的学习算法，包括线性回归和逻辑回归，它们能够有效地解决许多问题，但是当将它们应用到某些特定的机器学习应用时，会遇到**过拟合(over-fitting)**的问题，可能会导致它们效果很差。

在这段视频中，我将为你解释什么是过度拟合问题，并且在此之后接下来的几个视频中，我们将谈论一种称为**正则化(regularization)**的技术，它可以改善或者减少过度拟合问题。

![image-20200508163035221](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508163035221.png)

第一个模型是一个线性模型，欠拟合，不能很好地适应我们的训练集；第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据。我们可以看出，若给出一个新的值使之预测，它将表现的很差，是过拟合，虽然能非常好地适应我们的训练集但在新输入变量进行预测时可能会效果不好；而中间的模型似乎最合适。

分类问题中也存在这样的问题：

![image-20200508163058610](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508163058610.png)

就以多项式理解，x的次数越高，拟合的越好，但相应的预测的能力就可能变差。

问题是，如果我们发现了过拟合问题，应该如何处理？

1.丢弃一些不能帮助我们正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来帮忙（例如PCA）

2.正则化。 保留所有的特征，但是减少参数的大小。

### 7.2 代价函数

上面的回归问题中如果我们的模型是：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps9.jpg) 

我们可以从之前的事例中看出，正是那些高次项导致了过拟合的产生，所以如果我们能让这些高次项的系数接近于0的话，我们就能很好的拟合了。 

所以我们要做的就是在一定程度上减小这些参数θ 的值，这就是正则化的基本方法。我们决定要减少θ3和θ4的大小，我们要做的便是修改代价函数，在其中θ3和θ4设置一点惩罚。这样做的话，我们在尝试最小化代价时也需要将这个惩罚纳入考虑中，并最终导致选择较小一些的θ3和θ4

修改后的代价函数如下：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps11.jpg)

因为目标是要让代价函数最小，所以只能让θ3和θ4尽可能接近0。

假如我们有非常多的特征，我们并不知道其中哪些特征我们要惩罚，我们将对所有的特征进行惩罚，并且让代价函数最优化的软件来选择这些惩罚的程度。这样的结果是得到了一个较为简单的能防止过拟合问题的假设：![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps12.jpg)

其中λ又称为正则化参数（***\*Regularization Parameter\****）。 注：根据惯例，我们不对θ0进行惩罚。经过正则化处理的模型与原模型的可能对比如下图所示：

![image-20200508170003643](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508170003643.png)

如果选择的正则化参数λ过大，则会把所有的参数都最小化了，导致模型变成 ![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps13.jpg)，也就是上图中红色直线所示的情况，造成欠拟合。 

那为什么增加的一项![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps14.jpg) 可以使θ的值减小呢？ 

因为如果我们令λ的值很大的话，为了使代价函数尽可能的小，所有的θ的值（不包括θ0)都会在一定程度上减小。 

但若λ的值太大了，那么θ（不包括θ0)都会趋近于0，这样我们所得到的只能是一条平行于x轴的直线。 

所以对于正则化，我们要取一个合理的λ的值，这样才能更好的应用正则化。

### 7.3 正则化线性回归

对于线性回归的求解，我们之前推导了两种学习算法：一种基于梯度下降，一种基于正规方程。

正则化线性回归的代价函数为：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps16.jpg) 

如果我们要使用梯度下降法令这个代价函数最小化，因为我们未对进行正则化，所以梯度下降算法将分两种情形：

![image-20200508171556425](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508171556425.png)

对上面的算法中![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps17.jpg) 时的更新式子进行调整可得：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps18.jpg) 

可以看出，正则化线性回归的梯度下降算法的变化在于，每次都在原有算法更新规则的基础上令θ值减少了一个额外的值。

我们同样也可以利用正规方程来求解正则化线性回归模型，方法如下所示：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps19.jpg) 

图中的矩阵尺寸为(n+1)*(n+1)

若λ>0，必定可逆，解决了若m<n时，不可逆的问题。

### 7.4 正则化的逻辑回归模型

针对逻辑回归问题，我们在之前的课程已经学习过两种优化算法：我们首先学习了使用梯度下降法来优化代价函数J(θ)，接下来学习了更高级的优化算法，这些高级优化算法需要你自己设计代价函数J(θ)

![image-20200508173207841](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508173207841.png)

自己计算导数同样对于逻辑回归，我们也给代价函数增加一个正则化的表达式，得到代价函数：

![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps21.jpg)

要最小化该代价函数，通过求导，得出梯度下降算法为：

![image-20200508173231131](C:\Users\xuyingfeng\AppData\Roaming\Typora\typora-user-images\image-20200508173231131.png)

注：看上去同线性回归一样，但是知道![img](file:///C:\Users\XUYING~1\AppData\Local\Temp\ksohtml9816\wps22.jpg)，所以与线性回归不同



