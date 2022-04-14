# Second Week

## Multivariate linear regression 

$\begin{aligned} x_{j}^{(i)} &=\text { value of feature } j \text { in the } i^{\text {th }} \text { training example } \\ x^{(i)} &=\text { the input (features) of the } i^{\text {th }} \text { training example } \\ m &=\text { the number of training examples } \\ n &=\text { the number of features } \end{aligned}$

$h_{\theta}(x)=\left[\begin{array}{llll}\theta_{0} & \theta_{1} & \cdots & \theta_{n}\end{array}\right]\left[\begin{array}{c}x_{0} \\ x_{1} \\ \vdots \\ x_{n}\end{array}\right]=\theta^{T} x$

## Gradient Descent for Multiple Variables

$\begin{aligned} \theta_{0} &:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{0}^{(i)} \\ \theta_{1} &:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{1}^{(i)} \\ \theta_{2} &:=\theta_{2}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{2}^{(i)} \end{aligned}$

$\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{j}^{(i)} \quad$ for $\mathrm{j}:=0 \ldots \mathrm{n}$

## Feature Scaling

$$
x_{i}:=\frac{x_{i}-\mu_{i}}{s_{i}}
$$
Where $\mu_{i}$ is the average of all the values for feature (i) and $s_{i}$ is the range of values (max-min), or $s_{i}$ is the standard deviation.

## Normal Equation

$\theta = (X^TX)^{-1}X^Ty$

| Gradient Descent           | Normal Equation                               |
| :------------------------- | :-------------------------------------------- |
| Need to choose alpha       | No need to choose alpha                       |
| Needs many iterations      | No need to iterate                            |
| O ($kn^2$)                 | O($n^3$), need to calculate inverse of $X^TX$ |
| Works well when n is large | Slow if n is very large                       |

$y = X\theta$

### 线性代数最小二乘法

我们尝试将数据组$(x_1,y_1),...,(x_2,y_2)$

拟合到某个多项式$y = a_0+a_1x+a_2+x^2+...+a_nx^n$上

理想中，我们希望

$a_0+a_1x_1+...+a_nx_1^n = y_1$

$a_0+a_2x_1+...+a_nx_2^n = y_2$

​                        $\vdots$

$a_0+a_1x_n+...+a_nx_1=n^n = y_1$

因此我们建立

$\vec{b}=\left[\begin{array}{c}y_{1} \\ \vdots \\ y_{k}\end{array}\right]$
$X=\left[\begin{array}{ccccc}1 & x_{1} & x_{1}^{2} & \cdots & x_{1}^{n} \\ 1 & x_{2} & x_{2}^{2} & \cdots & x_{2}^{n} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{k} & x_{k}^{2} & \cdots & v_{k}^{n}\end{array}\right]$

$\vec{a} = \left[\begin{array}{c}a_{0} \\ \vdots \\ a_{n}\end{array}\right]$

现在只需要解方程$X\vec{a}=\vec{b}$

但基于数据组，$X\vec{a}=\vec{b}$可能矛盾，因此我们尝试寻找一个最佳拟合的多项式(即$\|X \vec{a}-\vec{b}\|$被最小化时的多项式)

根据正则系引理，我们看到 ![[公式]](https://www.zhihu.com/equation?tex=%7C%7CX%5Cvec+a-%5Cvec+b%7C%7C) 被最小化当且仅当$X^TX\vec{a} = X^T\vec{b}$

所以$\vec{a} = (X^TX)^{-1}X^T\vec{b}$

