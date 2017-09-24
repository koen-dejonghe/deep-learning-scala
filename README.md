
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
  val miniBatchSize = 16
  
   val (input, output) =
      Network.initialize(layout,
                         dimensions,
                         miniBatchSize,
                         regularization,
                         optimizer,
                         cost)
```

This will create a network with 2 hidden layers, each consisting of a linearity and a RELU nonlinearity, and an output layer.
The number of nodes in each of the layers is specified by the dimensions array. We will use the momentum optimizer with a learning rate of 0.3.
To evaluate the cost we use the softmax function, and regularization is set at 1e-4. 
The initializer returns references to the input and output actor, 
which are used for sending training and test sets to, and for monitoring.

```scala
  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz")

  input ! Forward(xTrain, yTrain)

```
To monitor progress, you could use a function like the following, 
which sends a Predict request for the test and training set through the network every 5 seconds,
and evaluates the accuracy and the cost (the latter only in case of a prediction on the training set).
```scala
  def monitor() = system.scheduler.schedule(5 seconds, 5 seconds) {

    implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

    (input ? Predict(xTest)).mapTo[Tensor].onComplete { d =>
      logger.info(s"accuracy on test set: ${accuracy(d.get, yTest)}")
    }

    (input ? Predict(xTrain)).mapTo[Tensor].onComplete { d =>
      val (c, _) = cost(d.get, yTrain)
      val a = accuracy(d.get, yTrain)
      logger.info(s"accuracy on training set: $a cost: $c")
    }
  }

  def accuracy(x: Tensor, y: Tensor): Double = {
    val m = x.shape(1)
    val p = numsca.argmax(x, 0)
    numsca.sum(p == y) / m
  }
```

### How does it work
A network is composed of gates (layers). Each gate is an actor in the actor system, and as such runs asynchronously in its own thread.

#### Forward pass
The training set is forwarded to the input actor. The input actor takes a random sample of size `miniBatchSize` from the training set, 
and forwards it to the first gate (which is supposed to be a linear gate). 
The activation function forwards it to the next gate, in this example, a RELU non-linearity. 
The RELU gate then forwards its activation to the next layer, again a linearity. 
And so on, until the output gate is reached. 
The output gate calculates the cost and the derivative of the cost using the provided cost function.

#### Backward pass
The derivative of the cost is fed back to the last layer, which calculates the gradient and passes this again to the gate before it.
And so on, until the input gate is reached.
At the input gate, a new sample is taken, and the process starts all over.

### Background

It seems like everything in machine learning must be written in Python these days.
Main culprit for this in my opinion is the excellent numpy library.
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
    val loss = numsca.sum(margins) / n

    val numPos = numsca.sum(margins > 0, axis = 1)
    val dx = numsca.zerosLike(x)
    dx.put(margins > 0, 1)
    dx.put(y, (ix, d) => d - numPos(ix.head))
    dx /= n

    (loss, dx)
  }
```

Numsca can be found [here]( https://github.com/koen-dejonghe/deep-learning-scala/tree/master/src/main/scala/numsca )

### Credit
Much of the stuff developed here is a result of some excellent courses I took, most notably: 

- CS231n: Convolutional Neural Networks for Visual Recognition
  - [Notes](http://cs231n.github.io/)
  - [Classroom videos](https://www.youtube.com/playlist?list=PL70hhrN6k0-CmnEhCnZLVP_0d9XH3edXW)
  - [Python code](https://github.com/koen-dejonghe/cs231n)

- Coursera's Deep Learning specialization 
  - [https://www.coursera.org/specializations/deep-learning]

