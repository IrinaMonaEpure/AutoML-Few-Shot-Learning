####################################################################
# Implement the two techniques Finetuning and Reptile
####################################################################

from networks import SineNetwork # the sine network that you should use
import torch.optim as optim
import torch.nn as nn
import torch
import time
import copy
import matplotlib.pyplot as plt

# A sine network can be used like this: net = SineNetwork() and called on inputs using net(input)

class Finetuning:
    def __init__(self):
        self.sine_network = SineNetwork()
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.001, momentum=0.9)
       
    def pretrain(self, batch): #batch should come from a non-episodic loader
        losses = []
        iters = []
        
        start_time = time.time()
        best_sine_network = copy.deepcopy(self.sine_network.state_dict())
        best_loss = 100
        crt_iter = 0
        x_train, y_train, _, _ = batch
        
        train_loss = 0
        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]
            self.optimizer.zero_grad() 
                
            # forward + tracking best model during training
            with torch.set_grad_enabled(True):
                output = self.sine_network(x)
                loss = self.sine_network.criterion(output, y)
                    
                # backward + optimize
                loss.backward()
                self.optimizer.step()
                
            train_loss += loss.item()
                
        epoch_train_loss = train_loss / len(x_train)
        losses.append(epoch_train_loss)
        iters.append(crt_iter)
        crt_iter += 1
            
        if epoch_train_loss<best_loss or best_loss==0:
            best_loss = epoch_train_loss
            best_sine_network = copy.deepcopy(self.sine_network.state_dict())
        
        train_time = time.time() - start_time
        self.sine_network.load_state_dict(best_sine_network)
        plt.plot(iters, losses)
        
        return epoch_train_loss, train_time
    
    def finetune(self, episode, epochs): # test loader must be episodic
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.1, momentum=0.9)
        start_time = time.time()
        total_train_loss = 0
        total_val_loss = 0
        
        x_support, y_support, x_query, y_query = episode
        initial_weights = copy.deepcopy(self.sine_network.state_dict())
        crt_sine_network = SineNetwork()
        crt_sine_network.load_state_dict(initial_weights)
            
        # Freeze all the network
        for param in crt_sine_network.parameters():
            param.requires_grad = False
                
        # Reset last layer
        crt_sine_network.model.update({"out": nn.Linear(64, 1)})
            
        for k in range(epochs):
            train_loss = 0
            
            for j in range(len(x_support)):
                x = x_support[j]
                y = y_support[j]
                self.optimizer.zero_grad() 
                        
                # forward
                with torch.set_grad_enabled(True):
                    output = crt_sine_network(x)
                    loss = crt_sine_network.criterion(output, y)
                            
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
                        
                train_loss += loss.item()
                    
            train_loss /= len(x_support)
            total_train_loss += train_loss
            val_loss = 0
                
        for i in range(len(x_query)):
            x = x_query[i]
            y = y_query[i]
            self.optimizer.zero_grad() 
                    
            # forward
            with torch.set_grad_enabled(False):
                output = crt_sine_network(x)
                loss = crt_sine_network.criterion(output, y)
                    
            val_loss += loss.item()
                
        val_loss /= len(x_query)
        finetune_time = time.time() - start_time
        
        return val_loss
        
        
class Reptile:
    def __init__(self):
        self.sine_network = SineNetwork()
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.0001, momentum=0.9)
        
    def train(self, batch, e): # tasks is a list of episodes
        start_time = time.time()
        losses = []
        iters = []
        j = len(batch) # batch size
        theta = copy.deepcopy(self.sine_network.state_dict()) # initial NN parameters
        theta_j = [] # list of parameter settings after SGD on theta
        
        # parse throught the batch, for every task in it do
        for task in batch:
            x_support, y_support, x_query, y_query = task
            train_loss = 0
            
            # train the network on a task, using both the support and query sets
            # support set
            for i in range(len(x_support)):
                x = x_support[i]
                y = y_support[i]
                self.optimizer.zero_grad() 
                    
                # forward + tracking best model during training
                with torch.set_grad_enabled(True):
                    output = self.sine_network(x)
                    loss = self.sine_network.criterion(output, y)
                        
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
            
            # query set
            for i in range(len(x_query)):
                x = x_query[i]
                y = y_query[i]
                self.optimizer.zero_grad() 
                    
                # forward + tracking best model during training
                with torch.set_grad_enabled(True):
                    output = self.sine_network(x)
                    loss = self.sine_network.criterion(output, y)
                        
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
                        
            # parameters after training on task j
            theta_j.append(copy.deepcopy(self.sine_network.state_dict()))
        
        # update theta at the end of the batch
        for t_j in theta_j:
            for (key,val) in t_j.items():
                theta[key] += (e/j) * (t_j[key] - theta[key])
        
        # load parameters from theta in the network
        self.sine_network.load_state_dict(theta)
        train_loss += loss.item()
        reptile_time = time.time() - start_time
        
        return loss, reptile_time
        
    def evaluate(self, episode, epochs):
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.1, momentum=0.9)
        
        # An average loss over one task is returned.
        x_support, y_support, x_query, y_query = episode
        initial_weights = copy.deepcopy(self.sine_network.state_dict())
        crt_sine_network = SineNetwork()
        crt_sine_network.load_state_dict(initial_weights)
        val_loss = 0
        train_loss = 0
        
        for k in range(epochs):
            for j in range(len(x_support)):
                x = x_support[j]
                y = y_support[j]
                self.optimizer.zero_grad() 
                    
                # forward
                with torch.set_grad_enabled(True):
                    output = crt_sine_network(x)
                    loss = crt_sine_network.criterion(output, y)
                        
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
                    
                train_loss += loss.item()
                
        train_loss /= len(x_support)
        train_loss /= epochs
            
        for i in range(len(x_query)):
            x = x_query[i]
            y = y_query[i]
                
            # forward
            with torch.set_grad_enabled(False):
                output = self.sine_network(x)
                loss = self.sine_network.criterion(output, y)
                
            val_loss += loss.item()
                
        val_loss /= len(x_query)
        
        return val_loss
        
        
class Maml:
    def __init__(self):
        self.sine_network = SineNetwork()
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.001, momentum=0.9)
        
    def train(self, batch, beta, del_theta): # tasks is a list of episodes
        start_time = time.time()
        losses = []
        iters = []
        j = len(batch) # batch size
        theta = copy.deepcopy(self.sine_network.state_dict()) # initial NN parameters
        theta_j = [] # list of parameter settings after SGD on theta
        
        # parse throught the batch, for every task in it do
        for task in batch:
            x_support, y_support, x_query, y_query = task
            train_loss = 0
            
            # train the network on a task, using only the support set
            for i in range(len(x_support)):
                x = x_support[i]
                y = y_support[i]
                self.optimizer.zero_grad() 
                    
                # forward + tracking best model during training
                with torch.set_grad_enabled(True):
                    output = self.sine_network(x)
                    loss = self.sine_network.criterion(output, y)
                        
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
            
            # Calculate the loss using the query set
            task_loss = 0
            for i in range(len(x_query)):
                x = x_query[i]
                y = y_query[i]
                self.optimizer.zero_grad() 
                    
                # forward + tracking best model during training
                with torch.set_grad_enabled(True):
                    output = self.sine_network(x)
                    loss = self.sine_network.criterion(output, y)
                    task_loss += loss.item()
                    
            task_loss /= len(x_query)
            losses.append(task_loss)
            # parameters after training on task j
            theta_j.append(copy.deepcopy(self.sine_network.state_dict()))
        
        # update theta at the end of the batch
        for j in range(len(theta_j)):
            t_j = theta_j[j]
            loss_j = losses[j]
            for (key,val) in t_j.items():
                theta[key] -= beta * del_theta * loss_j * t_j[key]
        
        # load parameters from theta in the network
        self.sine_network.load_state_dict(theta)
        train_loss += loss.item()
        maml_time = time.time() - start_time
        
        return loss, maml_time
        
    def evaluate(self, episode, epochs):
        self.optimizer = optim.SGD(self.sine_network.parameters(), lr=0.1, momentum=0.9)
        
        # An average loss over one task is returned.
        x_support, y_support, x_query, y_query = episode
        initial_weights = copy.deepcopy(self.sine_network.state_dict())
        crt_sine_network = SineNetwork()
        crt_sine_network.load_state_dict(initial_weights)
        val_loss = 0
        train_loss = 0
        
        for k in range(epochs):
            for j in range(len(x_support)):
                x = x_support[j]
                y = y_support[j]
                self.optimizer.zero_grad() 
                    
                # forward
                with torch.set_grad_enabled(True):
                    output = crt_sine_network(x)
                    loss = crt_sine_network.criterion(output, y)
                        
                    # backward + optimize
                    loss.backward()
                    self.optimizer.step()
                    
                train_loss += loss.item()
                
        train_loss /= len(x_support)
        train_loss /= epochs
            
        for i in range(len(x_query)):
            x = x_query[i]
            y = y_query[i]
                
            # forward
            with torch.set_grad_enabled(False):
                output = self.sine_network(x)
                loss = self.sine_network.criterion(output, y)
                
            val_loss += loss.item()
                
        val_loss /= len(x_query)
        
        return val_loss