# Deep Learning (Caffe) "Hello World" that computes squares in C++

This repository creates a very simple caffe deep learning model to compute squares of numbers between 1 & 100. 

This example can be simplified further if required.

I compiled this using the following command line on my environment. 
```
$ g++ -g -o train.out -DCPU_ONLY -I ../libs/caffe/distribute/include/ -L ../libs/caffe/distribute/lib/ square-train.cpp -lcaffe -lglog -lboost_system -lprotobuf;
```
