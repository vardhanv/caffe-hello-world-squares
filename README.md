# caffe Deep Learning "Hello World" that computes squares 

This repository creates a very simple caffe deep learning model to compute squares of numbers between 1 & 100. 
I found very little documentation that allows the creation of a simple hello world.

This example can be simplified further if required.

I compiled this using the following command line on my environment. 
'''
$ g++ -g -o train.out -DCPU_ONLY -I ../libs/caffe/distribute/include/ -L ../libs/caffe/distribute/lib/ square-train.cpp -lcaffe -lglog -lboost_system -lprotobuf;
'''
