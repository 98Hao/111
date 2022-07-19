# Fifth Week

## Cost Function

$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}$

For neural networks, it is going to be slightly more complicated:

$J(\Theta)=-\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K}\left[y_{k}^{(i)} \log \left(\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)\right]+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s l} \sum_{j=1}^{s l+1}\left(\Theta_{j, i}^{(l)}\right)^{2}$

## Backpropagation Algorithm

![image-20220503102702076](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220503102702076.png)

Given training set $\left\{\left(x^{(1)}, y^{(1)}\right) \cdots\left(x^{(m)}, y^{(m)}\right)\right\}$
- Set $\Delta_{i, j}^{(l)}:=0$ for all $(l, i, j)$, (hence you end up having a matrix full of zeros)
For training example $\mathrm{t}=1$ to $\mathrm{m}$ :

1. $\operatorname{Set} a^{(1)}:=x^{(t)}$

2. Perform forward propagation to compute $a^{(l)}$ for $l=2,3, \ldots, L$

   ![image-20220503102949747](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220503102949747.png)

   3. Using $y^{(t)}$, compute $\delta^{(L)}=a^{(L)}-y^{(t)}$
   Where $L$ is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:
   4. Compute $\delta^{(L-1)}, \delta^{(L-2)}, \ldots, \delta^{(2)}$ using $\delta^{(l)}=\left(\left(\Theta^{(l)}\right)^{T} \delta^{(l+1)}\right) \cdot * a^{(l)} \cdot *\left(1-a^{(l)}\right)$
   The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the theta matrix of layer $l$. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function $g$ evaluated with the input values given by $z^{(l)}$.
   The g-prime derivative terms can also be written out as:
   $$
   g^{\prime}\left(z^{(l)}\right)=a^{(l)} \cdot *\left(1-a^{(l)}\right)
   $$
   5. $\Delta_{i, j}^{(l)}:=\Delta_{i, j}^{(l)}+a_{j}^{(l)} \delta_{i}^{(l+1)}$ or with vectorization, $\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}\left(a^{(l)}\right)^{T}$ Hence we update our new $\Delta$ matrix.

   - $D_{i, j}^{(l)}:=\frac{1}{m}\left(\Delta_{i, j}^{(l)}+\lambda \Theta_{i, j}^{(l)}\right)$, if $\mathrm{j \neq 0.}$
   - $D_{i, j}^{(l)}:=\frac{1}{m} \Delta_{i, j}^{(l)}$ If $\mathrm{j}=0$
     The capital-delta matrix $D$ is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac{\partial}{\partial \Theta_{i j}^{(l)}} J(\Theta)=D_{i j}^{(l)}$

## Random Initializzation

![image-20220506102748799](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220506102748799.png)

```matlab
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

