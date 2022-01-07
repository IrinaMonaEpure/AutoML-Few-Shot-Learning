####################################################################
# Feel free to change anything in this file as you wish
####################################################################

import argparse
import numpy as np
import os
import torch.optim as optim
import matplotlib.pyplot as plt

from algorithms import Finetuning, Reptile, Maml
from data_loader import SineLoader
from networks import SineNetwork

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


####################################################################
# You can also change these defaults. They are for debugging and may 
# be non-optimal
####################################################################
parser=argparse.ArgumentParser()
parser.add_argument('--folder', default="./data/", help="Data folder to store the dataset") # do not change this line!
parser.add_argument('--num_ways', default=5, help="Number of classes ")
parser.add_argument('--num_shots_support', default=5, help="Number of examples per class in the support set")
parser.add_argument('--num_shots_query', default=32, help="Number of examples per class in the query set")
parser.add_argument('--train_iters', default=100, help="Number of training iterations")
parser.add_argument('--train_batch_size', default=16, help="Batch size for the fine-tuning method during training time")
parser.add_argument('--test_batch_size', default=16, help="Batch size for the fine-tuning method during test time")
args = parser.parse_args()

if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

data_loader = SineLoader(k=args.num_shots_support, k_test=args.num_shots_query)

# Algorithm classes
ft = Finetuning()
reptile = Reptile()
maml = Maml()

# Some lists that will be useful for plotting the results
ft_losses_train, reptile_losses_train, maml_losses_train = [], [], []
ft_time_train, reptile_time_train, maml_time_train = [], [], []
ft_losses_val, reptile_losses_val, maml_losses_val = [], [], []
ft_losses_test, reptile_losses_test, maml_losses_test = [], [], []
ft_epochs, tasks = [], []

# We need a non-episodic set of data for pretraining
train_loader_normal = data_loader.generator(episodic=False, batch_size=args.train_batch_size, mode="train", reset_ptr=True)
pretrain_data = next(train_loader_normal)

# We also need an episodic train loader for Reptile and Maml
train_loader_episodic = data_loader.generator(episodic=True, batch_size=args.train_batch_size, mode="train", reset_ptr=True)

# Finally, we will need episodic validation and test loaders
val_loader = data_loader.generator(episodic=True, batch_size=args.test_batch_size, mode="val", reset_ptr=True)
test_loader = data_loader.generator(episodic=True, batch_size=args.test_batch_size, mode="test", reset_ptr=True)

# Set some parameters for reptile and maml
e = 0.5
beta = 0.1
del_theta = 0.1

# Batch of tasks for reptile and Maml
batch = []
it = 0

for ft_epoch in range(300):
    # Pretrain the finetunuing network and save some data about performance
    (ft_loss, ft_time) = ft.pretrain(pretrain_data)
    ft_losses_train.append(ft_loss)
    ft_time_train.append(ft_time)
    ft_epochs.append(ft_epoch)
    
    
plt.plot(ft_epochs, ft_losses_train)
plt.title("Finetuning Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Train three networks using different algorithms for the same number of classes
for it in range(70000):
    # Get data from the episodic train loader
    episode = next(train_loader_episodic)
        
    # Fill out a batch of tasks for Reptile and Maml
    batch.append(episode)
        
    # When the batch is full, train Reptile and Maml and save some performance data
    if len(batch) == args.train_batch_size:
        (reptile_loss, reptile_time) = reptile.train(batch, e)
        (maml_loss, maml_time) = maml.train(batch, beta, del_theta)
        reptile_losses_train.append(reptile_loss)
        maml_losses_train.append(maml_loss)
        reptile_time_train.append(reptile_time)
        maml_time_train.append(maml_time)
        batch = []
        tasks.append(it)
    it += 1   

plt.plot(tasks, reptile_losses_train)
plt.title("Reptile Training")
plt.xlabel("Tasks")
plt.ylabel("Loss")
plt.show()

plt.plot(tasks, maml_losses_train)
plt.title("Maml Training")
plt.xlabel("Tasks")
plt.ylabel("Loss")
plt.show()

print("Time needed for pretraining: ", sum(ft_time_train))
print("Time needed for training Reptile: ", sum(reptile_time_train))
print("Time needed for training Maml: ", sum(maml_time_train))
print("Final training loss for Finetuning: ", ft_losses_train[-1])
print("Final training loss for Reptile: ", reptile_losses_train[-1])
print("Final training loss for Maml: ", maml_losses_train[-1])
    
# Validation on all episodic data from the val_loader
# Use this opportunity to observe what are the best values for e, beta and del_theta
for i in range(1000):
    # Get a task from the val_loader
    episode = next(val_loader)
        
    # Train and validate the three networks and save some data about performance
    ft_loss = ft.finetune(episode, epochs=2)
    reptile_loss = reptile.evaluate(episode, epochs=2)
    maml_loss = maml.evaluate(episode, epochs=2)
    ft_losses_val.append(ft_loss)
    reptile_losses_val.append(reptile_loss)
    maml_losses_val.append(maml_loss)
    
print("Average validation loss for Finetuning: ", sum(ft_losses_val)/len(ft_losses_val))
print("Average validation loss for Reptile: ", sum(reptile_losses_val)/len(reptile_losses_val))
print("Average validation loss for Maml: ", sum(maml_losses_val)/len(maml_losses_val))

# Testing on all episodic data from the test loader

for i in range(1000):
    # Get a task from the test_loader
    episode = next(test_loader)
        
    # Train and validate the three networks and save some data about performance
    ft_loss = ft.finetune(episode, epochs=2)
    reptile_loss = reptile.evaluate(episode, epochs=2)
    maml_loss = maml.evaluate(episode, epochs=2)
    ft_losses_test.append(ft_loss)
    reptile_losses_test.append(reptile_loss)
    maml_losses_test.append(maml_loss)
    
print("Average test loss for Finetuning: ", sum(ft_losses_test)/len(ft_losses_val))
print("Average test loss for Reptile: ", sum(reptile_losses_test)/len(reptile_losses_val))
print("Average test loss for Maml: ", sum(maml_losses_test)/len(maml_losses_val))
