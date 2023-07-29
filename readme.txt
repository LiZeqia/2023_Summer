#This folder is used to record the relevent practice content of the summer trainning.

"----For task1 ----"
#The goal of the task is to build a classifier based on the Cifar10 dataset.
#I build three classifier: 3-layer DNN, 8-layer DNN and CNN, the result shows that CNN is the most useful.
#I experimented with different learning rates and found that learning rates too small and too large were not good. Small learning tate makes the model fall into local optimality, and big learning rate makes the model can not get into the optimality.
#And also I test different batch_size, the result also shows that the size of batch can not be too large or small.

"----For task2 ----"
#The goal of the task is to probe the function of residual block.
#I implement the residual network and non-residual network based on the resnet_18.
#The result shows that the model with residual block is trained faster and the accuracy is higher.