# First Week

## Machine learning algorithms:

### Supervised learning 

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220410113522635.png" alt="image-20220410113522635" style="zoom:50%;" />

<img src="C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20220410113554645.png" alt="image-20220410113554645" style="zoom:50%;" />

从标记的训练数据来推断一个功能的机器学习任务

- regression problem（回归问题）
- classification problem (分类问题)

### Unsupervised learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

## Cost Function

$J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(\hat{y}_{i}-y_{i}\right)^{2}=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x_{i}\right)-y_{i}\right)^{2}$

Hypothesis: $\quad h_{\theta}(x)=\theta_{0}+\theta_{1} x$
Parameters: $\quad \theta_{0}, \theta_{1}$
Cost Function: $\quad J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$
Goal: $\quad \operatorname{minimize}_{\theta_{0}, \theta_{1}} J\left(\theta_{0}, \theta_{1}\right)$

## Gradient Descent

$\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J\left(\theta_{0}, \theta_{1}\right)\quad(j = 0,1)$

$\theta_{0}:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)$
$\theta_{1}:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x^{(i)}$



