
# 第三讲：三维空间刚体运动

由于 github 的 markdown 不能渲染公式，可以到我的 [notion](https://secret-cloche-bec.notion.site/92668a49772747febaacd29cdd372ebd) 页面查看

> **主要目标： 1. 理解三维空间的刚体运动描述方式：旋转矩阵、变化矩阵、四元数和欧拉角 。2. 掌握 Eigen 库的矩阵、几何模块的使用方法。**

这一讲主要介绍视觉 SLAM 的基本问题之一：如何描述刚体在三维空间中的运动？

# 旋转矩阵

## 点、向量和坐标系

**点** 就是空间中的基本元素，没有长度，没有提及。把两个点连接起来就构成了 **向量**。不要将坐标和向量的概念混淆，只有当给定坐标系时，才可以讨论在向量的坐标。

使用线性代数来解释，首先定义一组基（ $\boldsymbol{e}_1,\boldsymbol{e}_2,\boldsymbol{e}_3$ ），那么任意向量 $\boldsymbol{a}$ 在这组基下就有一个 **坐标**：

$$
\boldsymbol{a}=\left[\boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \boldsymbol{e}_{3}\right]\left[\begin{array}{l}a_{1} \\a_{2} \\a_{3}\end{array}\right]=a_{1} \boldsymbol{e}_{1}+a_{2} \boldsymbol{e}_{2}+a_{3} \boldsymbol{e}_{3}
$$

这里的 $[a_1,a_2,a_3]^T$ 称为 $\boldsymbol{a}$ 在此基下的坐标。 

向量之间的运算有数乘、加法、减法、内积、外积。其中内积和外积稍微复杂一点。

![Untitled](%E7%AC%AC%E4%B8%89%E8%AE%B2%EF%BC%9A%E4%B8%89%E7%BB%B4%E7%A9%BA%E9%97%B4%E5%88%9A%E4%BD%93%E8%BF%90%E5%8A%A8%209e5643ad2598493b82d27260dbaafe29/Untitled.png)

$\boldsymbol{a,b}$ 之间的内积可以写成：

$$\boldsymbol{a} \cdot \boldsymbol{b}=\boldsymbol{a}^{\mathrm{T}} \boldsymbol{b}=\sum_{i=1}^{3} a_{i} b_{i}=|\boldsymbol{a}||\boldsymbol{b}| \cos \langle\boldsymbol{a}, \boldsymbol{b}\rangle$$

其中 $\langle\boldsymbol{a}, \boldsymbol{b}\rangle$ 之向量 $\boldsymbol{a,b}$ 的夹角。内积可以描述向量间的投影关系。

![Untitled](%E7%AC%AC%E4%B8%89%E8%AE%B2%EF%BC%9A%E4%B8%89%E7%BB%B4%E7%A9%BA%E9%97%B4%E5%88%9A%E4%BD%93%E8%BF%90%E5%8A%A8%209e5643ad2598493b82d27260dbaafe29/Untitled%201.png)

内积在以前的学习中还经常用到过，但是外积又被称为**楔积**（Wedge Product）我就了解的比较少了。外积定义如下：

$$\boldsymbol{a} \times \boldsymbol{b}=\left\|\begin{array}{lll}e_{1} & e_{2} & e_{3} \\a_{1} & a_{2} & a_{3} \\b_{1} & b_{2} & b_{3}\end{array}\right\|=\left[\begin{array}{c}a_{2} b_{3}-a_{3} b_{2} \\a_{3} b_{1}-a_{1} b_{3} \\a_{1} b_{2}-a_{2} b_{1}\end{array}\right]=\left[\begin{array}{ccc}0 & -a_{3} & a_{2} \\a_{3} & 0 & -a_{1} \\-a_{2} & a_{1} & 0\end{array}\right] \boldsymbol{b} \stackrel{\text { def }}{=} \boldsymbol{a}^{\wedge} \boldsymbol{b} $$

外积的结果是一个向量，它的方向垂直于这两个向量，大小为 $|\boldsymbol{a}||\boldsymbol{b}|sin\langle\boldsymbol{a}, \boldsymbol{b}\rangle$ , 是两个向量张成的四边形的有向面积。 $\boldsymbol{a}^\wedge$ 可以被理解为一个反对称矩阵，那么 $\boldsymbol{a}^\wedge\boldsymbol{b}$ 可以被理解为矩阵于向量的乘法。向量的反对称矩阵是一一对应的：

$$\boldsymbol{a}^\wedge=[\begin{array}{ccc}0 & -a_3&a_2\\a_3 &0&a_1\\-a_2&a_1&0\end{array}]$$

## 坐标系间的欧式变换

![坐标变换： 对于同一个向量 $\boldsymbol{p}$ , 它在移动坐标系和世界坐标系下面的坐标是不同的。这个变化关系可以使用变换矩阵 $T$ 来描述。](%E7%AC%AC%E4%B8%89%E8%AE%B2%EF%BC%9A%E4%B8%89%E7%BB%B4%E7%A9%BA%E9%97%B4%E5%88%9A%E4%BD%93%E8%BF%90%E5%8A%A8%209e5643ad2598493b82d27260dbaafe29/Untitled%202.png)

坐标变换： 对于同一个向量 $\boldsymbol{p}$ , 它在移动坐标系和世界坐标系下面的坐标是不同的。这个变化关系可以使用变换矩阵 $T$ 来描述。

考虑运动的机器人时，一般会考虑使用两个坐标系，一个是惯性坐标系（也被称为世界坐标系），另一个是机器人或者相机的移动坐标系。一般，惯性坐标系是固定不动的，可以使用 $x_W,y_W,z_W$ 表示。顾名思义，移动坐标系是会移动的采用 $x_C,y_C,z_z$ 表示。相机视野中的某一个向量 $\boldsymbol{p}$ 在两个坐标系中的坐标是不同的，在惯性坐标系中，它的坐标为 $\boldsymbol{p}_W$。而在移动坐标系中，则为 $\boldsymbol{p}_C$ 。通过取移动坐标系的坐标，然后通过机器人的位姿 **变换** 就可以得到世界坐标系的坐标。接下来将通过数学含义来描述这种变化关系。

由于机器人是刚体，在移动过程中，同一个向量不会在不同的坐标系下长度和夹角不会发生变化。故移动坐标系和世界坐标系只差一个 **欧式变换。**

欧式变换由旋转和平移组成。旋转可以通过一个旋转矩阵 $\boldsymbol{R}$ 来表示，旋转矩阵是正交矩阵(正交矩阵的逆矩阵和其转置一样)，行列式为 1 ，为基向量的余弦值，所以旋转矩阵又被称为方向余弦矩阵。

假设向量 $\boldsymbol{a}$ 在世界坐标系为 $[a_1,a_2,a_3]$ ，移动坐标系为 $[a'_1,a'_2,a'_3]$ 。这样就有下列表示：

$$\boldsymbol{a}=\boldsymbol{Ra'}$$

欧式变化除了旋转还有平移 $\boldsymbol{t}$ ，那完整的欧式变化就可以使用下面的表示：

$$⁍$$

一般实际中，将坐标系 2 转化为 坐标系 1 会采用下面的数学形式表达：

$$\boldsymbol{a}_1=\boldsymbol{R}_{12}\boldsymbol{a}_{2}+\boldsymbol{t}_{12}$$

这里的 $\boldsymbol{R}_{12}$ 是指将坐标系 2 的向量转化到坐标系 1 中。 $\boldsymbol{t}_{12}$ 则表示为坐标系 1 原点到 坐标系 2 原点的向量。

## 变换矩阵和齐次坐标

如果称一个数学函数 $L(x)$ 为线性的，可以是指：

1. 是个只拥有一个[变数](https://zh.wikipedia.org/wiki/%E8%AE%8A%E6%95%B8)的一阶[多项式函数](https://zh.wikipedia.org/wiki/%E5%A4%9A%E9%A1%B9%E5%BC%8F%E5%87%BD%E6%95%B0)，即是可以表示为 $L(x)=kx+b$ 的形式（其中 $k,b$ 为[常数](https://zh.wikipedia.org/wiki/%E5%B8%B8%E6%95%B0)）。
2. $L(x)$具有以下两个性质：
    1. [可加性](https://zh.wikipedia.org/wiki/%E5%8F%AF%E5%8A%A0%E6%80%A7)： $L(x+t)=L(x)+L(t)$
    2. 一次齐次性：  $L(mx)=mL(x)$

一般高等数学中，需要遵守定义 2。 

通过上面的定义，我们可以很简单的得出上面的欧式变换公式不是线性的，如果我们要经过多次坐标变化的话，表达起来就比较繁琐，所以下面引入一个数学技巧，在三维向量末尾添加 1，将其变成四维向量，称为齐次坐标，这样我们就可以将旋转和平移写到同一个矩阵，这一个矩阵就被称为 $T$ 变换矩阵。

$$\left[\begin{array}{l}\boldsymbol{a}^{\prime} \\1\end{array}\right]=\left[\begin{array}{ll}\boldsymbol{R} & \boldsymbol{t} \\\boldsymbol{0}^{\mathrm{T}} & 1\end{array}\right]\left[\begin{array}{l}\boldsymbol{a} \\1\end{array}\right] \stackrel{\text { def }}{=} \boldsymbol{T}\left[\begin{array}{l}\boldsymbol{a} \\1\end{array}\right]$$