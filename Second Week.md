# Second Week

# Multivariate linear regression 

$\begin{aligned} x_{j}^{(i)} &=\text { value of feature } j \text { in the } i^{\text {th }} \text { training example } \\ x^{(i)} &=\text { the input (features) of the } i^{\text {th }} \text { training example } \\ m &=\text { the number of training examples } \\ n &=\text { the number of features } \end{aligned}$

$h_{\theta}(x)=\left[\begin{array}{llll}\theta_{0} & \theta_{1} & \cdots & \theta_{n}\end{array}\right]\left[\begin{array}{c}x_{0} \\ x_{1} \\ \vdots \\ x_{n}\end{array}\right]=\theta^{T} x$

## Gradient Descent for Multiple Variables

$\begin{aligned} \theta_{0} &:=\theta_{0}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{0}^{(i)} \\ \theta_{1} &:=\theta_{1}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{1}^{(i)} \\ \theta_{2} &:=\theta_{2}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{2}^{(i)} \end{aligned}$

$\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \cdot x_{j}^{(i)} \quad$ for $\mathrm{j}:=0 \ldots \mathrm{n}$

