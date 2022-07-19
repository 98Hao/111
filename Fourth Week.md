# Fourth Week

## Model Representation I

$\left[x_{0} x_{1} x_{2}\right] \rightarrow[\quad] \rightarrow h_{\theta}(x)$

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes $a^2_0 \cdots a^2_n$ and call them "activation units."

$a_{i}^{(j)}=$ "activation" of unit $i$ in layer $j$
$\Theta^{(j)}=$ matrix of weights controlling function mapping from layer $j$ to layer $j+1$
If we had one hidden layer, it would look like:
$$
\left[x_{0} x_{1} x_{2} x_{3}\right] \rightarrow\left[a_{1}^{(2)} a_{2}^{(2)} a_{3}^{(2)}\right] \rightarrow h_{\theta}(x)
$$
The values for each of the "activation" nodes is obtained as follows:
$$
\begin{array}{r}
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\
a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\
a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\
h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)
\end{array}
$$
![image-20220426102942285](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220426102942285.png)

## Model Representation II

$a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right)$
$a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right)$
$a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right)$
$h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)$
$$
\begin{aligned}
&a_{1}^{(2)}=g\left(z_{1}^{(2)}\right) \\
&a_{2}^{(2)}=g\left(z_{2}^{(2)}\right) \\
&a_{3}^{(2)}=g\left(z_{3}^{(2)}\right)
\end{aligned}
$$
In other words, for layer $j=2$ and node $k$, the variable $z$ will be:
$$
z_{k}^{(2)}=\Theta_{k, 0}^{(1)} x_{0}+\Theta_{k, 1}^{(1)} x_{1}+\cdots+\Theta_{k, n}^{(1)} x_{n}
$$
The vector representation of $x$ and $z^{j}$ is:
$$
x=\left[\begin{array}{c}
x_{0} \\
x_{1} \\
\cdots \\
x_{n}
\end{array}\right] z^{(j)}=\left[\begin{array}{c}
z_{1}^{(j)} \\
z_{2}^{(j)} \\
\cdots \\
z_{n}^{(j)}
\end{array}\right]
$$
Setting $x=a^{(1)}$, we can rewrite the equation as:
$$
z^{(j)}=\Theta^{(j-1)} a^{(j-1)}
$$

$$
a^{(j)} = g(z^{(j)})
$$

$$
z^{(j+1)} = \Theta^{(j)}a^{(j)}
$$

$$
h_{\Theta}(x)=a^{(j+1)}=g\left(z^{(j+1)}\right)
$$

