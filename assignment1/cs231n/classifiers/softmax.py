import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  trian_num = X.shape[0]
  class_num = W.shape[1]
  for i in range(trian_num):
    scores = X[i].dot(W)
    max_score = np.max(scores)
    scores -= max_score 
    sum_class_e = 0.0
    for j in scores:
      sum_class_e += np.exp(j)
    loss += -np.log(np.exp(scores[y[i]]) / sum_class_e)

    for j in range(class_num):
      p = np.exp(scores[j])/sum_class_e
      dW[:, j] += (p-(j==y[i])) * X[i, :]


  loss = loss / trian_num
  dW = dW / trian_num
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  trian_num = X.shape[0]
  class_num = W.shape[1]
  
  scores = X.dot(W)
  max_score = np.max(scores, axis=1)
  scores -= max_score[:, np.newaxis]
  exp_scores = np.exp(scores)
  sum_class_e = np.sum(exp_scores, axis=1)
  loss += np.sum(-np.log(exp_scores[range(trian_num), y] / sum_class_e))
  loss /= trian_num
  loss += 0.5 * reg * np.sum(W * W)

  
  p = exp_scores / sum_class_e[:, np.newaxis]
  ind = np.zeros(p.shape)
  ind[range(trian_num), y] = 1
  #p[:, y] -= 1
  dW = np.dot(X.T, (p-ind))
  dW /= trian_num
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

