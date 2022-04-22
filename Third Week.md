# Third Week

## Logistic Function

$h_{\theta}(x) = g(\theta^Tx)$

$z = \theta^Tx$

$g(z) = \frac{1}{1+e^{-z}}$

$h_{\theta}(x) = P(y=1|x;\theta) = 1 - P(y=0|x;\theta)$

$P(y=0|x;\theta)+P(y=1|x;\theta) = 1$

$y = 1 \rightarrow g(z)\geq0.5 \rightarrow z\geq0$

$y =  \rightarrow g(z)<0.5 \rightarrow z<0$

## Cost Function

$\begin{array}{ll}J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \operatorname{Cost}\left(h_{\theta}\left(x^{(i)}\right), y^{(i)}\right) & \\ \operatorname{Cost}\left(h_{\theta}(x), y\right)=-\log \left(h_{\theta}(x)\right) & \text { if } \mathrm{y}=1 \\ \operatorname{Cost}\left(h_{\theta}(x), y\right)=-\log \left(1-h_{\theta}(x)\right) & \text { if } \mathrm{y}=0\end{array}$

![image-20220416222443241](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220416222443241.png)

![image-20220416222500857](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220416222500857.png)

$\operatorname{Cost}\left(h_{\theta}(x), y\right)=0$ if $h_{\theta}(x)=y$
$\operatorname{Cost}\left(h_{\theta}(x), y\right) \rightarrow \infty$ if $y=0$ and $h_{\theta}(x) \rightarrow 1$
$\operatorname{Cost}\left(h_{\theta}(x), y\right) \rightarrow \infty$ if $y=1$ and $h_{\theta}(x) \rightarrow 0$

## Gradient Descent

$Cost(h_{\theta}(x),y) =  -y\log{h_{\theta}(x)}-(1-y)\log(1-h_{\theta}(x))$

entire cost function:

$J(\theta)=\frac{1}{m} \sum_{i=1}^{m}[y^{(i)}\log(h_{\theta}(x^{(i)}))+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))]$

$\theta:=\theta-\frac{\alpha}{m}X^T(g(X\theta)-\vec{y})$

## Advanced Optimization

We first need to provide a function that evaluates the following two functions for a given input value θ:

$J\theta$

$\frac{\partial}{\partial \theta_{j}} J(\theta)$

We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```



## Multiclass Classification: One-vs-all

$y \in\{0,1 \ldots n\}$
$h_{\theta}^{(0)}(x)=P(y=0 \mid x ; \theta)$
$h_{\theta}^{(1)}(x)=P(y=1 \mid x ; \theta)$
$\cdots$
$h_{\theta}^{(n)}(x)=P(y=n \mid x ; \theta)$
prediction $=\max _{i}\left(h_{\theta}^{(i)}(x)\right)$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

![image-20220418113250874](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220418113250874.png)

## Modify Cost Function

Say we wanted to make the following function more quadratic:
$$
\theta_{0}+\theta_{1} x+\theta_{2} x^{2}+\theta_{3} x^{3}+\theta_{4} x^{4}
$$
We'll want to eliminate the influence of $\theta_{3} x^{3}$ and $\theta_{4} x^{4}$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:
$$
\min _{\theta} \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+1000 \cdot \theta_{3}^{2}+1000 \cdot \theta_{4}^{2}
$$
![image-20220422091320856](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220422091320856.png)

$\min _{\theta} \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}$

# Regularized Linear Regression

### Gradient Descent

Repeat \{
$$
\begin{aligned}
&\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{0}^{(i)} \\
&\theta_{j}:=\theta_{j}-\alpha\left[\left(\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\right)+\frac{\lambda}{m} \theta_{j}\right] \quad j \in\{1,2 \ldots n\}
\end{aligned}
$$
}

### Normal Equation

$$
\theta=\left(X^{T} X+\lambda \cdot L\right)^{-1} X^{T} y
$$
where $L=$$\left[\begin{array}{lllll}0 & & & & \\ & 1 & & & \\ & & 1 & & \\ & & & \ddots & \\ & & & & 1\end{array}\right]$

# Regularized Logistic Regression

$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}$

![image-20220422091658464](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220422091658464.png)

