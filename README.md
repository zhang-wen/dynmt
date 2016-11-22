# Dynmt

A machine translation system using LSTM attention model

==================
this model is based on [dynet](https://github.com/clab/dynet) neural network library which is 
implemented by Carnegie Mellon University and many other developers. the low-layer code is written 
by using C++, low-layer functions are developed based on [Eigen](http://eigen.tuxfamily.org), which 
is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

not only they develop the C++ interface which is usually called [cnn](http://github.com/clab/cnn-v1), 
but also they implement the python interface [pycnn](http://github.com/clab/cnn-v1), and dynet is 
actually similar with pycnn.
