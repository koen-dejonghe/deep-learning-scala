
# Deep Learning with Scala

This project contains a blueprint for easy development of deep neural networks with Scala and Akka.
Focus is on elegance and simplicity rather than performance. 
Still it has some attractive potential, such as asynchronous network layers, which could be deployed on a cluster of machines.

### Building a neural net
Building a net is as simple as specifying the layout of the network, 
the size of the layers, and some hyperparameters.
```scala
  val layout = (Linear + Relu) * 2
  val dimensions = Array(784, 50, 10)

  def optimizer = Momentum(learningRate = 0.3)

  val cost: CostFunction = softmaxCost

  val regularization = 1e-4

  val numIterations = 500000
  val miniBatchSize = 16
  
   val (input, output) =
      Network.initialize(system,
                         layout,
                         dimensions,
                         miniBatchSize,
                         regularization,
                         optimizer,
                         cost)
```




--- 




* python vs scala
* no deep learning frameworks for scala
* numsca
* async neural net framework with akka
  * pros:
    * monitoring & tuning of individual layers by means of messaging
    * monitoring & tuning of network by means of messaging
    * inception during training, so possibility for endless training
    * easy integration in play framework
    * brain also works async
    * possibility to deploy in cluster
    * allows for easy and crazy experiments 

* courses

Scala implementation of assignments of some courses:

- CS231n: Convolutional Neural Networks for Visual Recognition
  - http://cs231n.github.io/
  - Python code is at: https://github.com/koen-dejonghe/cs231n

- Coursera's Deep Learning specialization 
  - https://www.coursera.org/specializations/deep-learning


Everything in ML must be written in Python these days it seems.
Main culprit for this is in my opinion the excellent numpy library.
The numsca package is my futile attempt to mimic some of its most useful functionality.
It's a thin wrapper around the nd4j N-Dimensional Arrays for Java library (http://nd4j.org/)

I think the result is quite elegant.
For example, check out the following code:

In Python with numpy:

```python
    def svm_loss(x, y):
     svm_loss(x, y):
        """
        Computes the loss and gradient using for multiclass SVM classification.
    
        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
          class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
          0 <= y[i] < C
    
        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """

        N = x.shape[0]
        correct_class_scores = x[np.arange(N), y]
        margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
        margins[np.arange(N), y] = 0
        loss = np.sum(margins) / N

        num_pos = np.sum(margins > 0, axis=1)
        dx = np.zeros_like(x)
        dx[margins > 0] = 1
        dx[np.arange(N), y] -= num_pos
        dx /= N

        return loss, dx
```

In Scala with numsca:


```scala
  def svmLoss(x: Tensor, y: Tensor): (Double, Tensor) = {
    val n = x.shape(0)
    val correctClassScores = x(y)
    val margins = numsca.maximum(x - correctClassScores + 1.0, 0.0)
    margins.put(y, 0)
    val loss = numsca.sum(margins).squeeze() / n

    val numPos = numsca.sum(margins > 0, axis = 1)
    val dx = numsca.zerosLike(x)
    dx.put(margins > 0, 1)
    dx.put(y, (ix, d) => d - numPos(ix.head))
    dx /= n

    (loss, dx)
  }
```

Numsca can be found at:

https://github.com/koen-dejonghe/deep-learning-scala/tree/master/src/main/scala/numsca
