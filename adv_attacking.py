# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:50:50 2021

@author: Alessandro Padella
"""

# Codes inspired to codes at https://jaketae.github.io/study/fgsm/ and
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from flags import gen_image, get_flags
from models_real import MNISTModel
from torch.optim import SGD, Adam
from torchvision import datasets, transforms

curr_dir = 'C:/Users/padel/OneDrive/Desktop/improved_contrastive_divergence-master'
os.chdir(curr_dir)

plt.rcParams['image.cmap'] = 'gray'


# Get functions and flags
FLAGS = get_flags()


# Set the hyperparams
def get_hparams():
    itr = 2
    kl = True
    FLAGS.kl = kl
    kl_str = (not kl)*'_nokl'
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = f'models/model_fd{itr}{kl_str}.pth'
    use_cuda = False
    return itr, kl, kl_str, epsilons, pretrained_model, use_cuda


itr, kl, kl_str, epsilons, pretrained_model, use_cuda = get_hparams()

# Import the test_loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/mnist', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True)

# Check availability of the GPU
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (
    use_cuda and torch.cuda.is_available()) else "cpu")


# Load the model
# Network
def get_pretrained_model(pretrained_model, FLAGS):
    FLAGS.filter_dim = itr
    mnist_model = MNISTModel(FLAGS)
    parameters = []
    parameters.extend(list(mnist_model.parameters()))
    optimizer = Adam(parameters, lr=FLAGS.lr, betas=(0.0, 0.9), eps=1e-8)
    checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
    mnist_model.load_state_dict(checkpoint['model_state_dict_0'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mnist_model.eval()
    return mnist_model


mnist_model = get_pretrained_model(pretrained_model, FLAGS)


# Have to evaluate the model, have to train (and test) the model with the read-out after one iteration

# Create the class
class lm_torch(nn.Module):
    def __init__(self, itr=2, kl=True):
        super(lm_torch, self).__init__()
        self.kl = kl
        self.itr = itr
        self.kl_str = (not self.kl)*'_nokl'
        self.lin = nn.Linear(128*self.itr, 10, bias=True)
        self.lin.load_state_dict(torch.load(
            f'models/lin_model_torch_fd{self.itr}{self.kl_str}_sd.pth'))
        # Have to change the name obv

    def forward(self, x):
        x = self.lin.forward(x)
        return x


# Mix the classes
class model_complete(nn.Module):
    def __init__(self, lm_torch, mnist_model):
        super(model_complete, self).__init__()
        self.lm_torch = lm_torch
        self.mnist_model = mnist_model

    def forward(self, image):
        x = self.mnist_model.forward(image, latent=None, read_out=True)
        x = self.lm_torch.forward(x)
        return x


lm_torch = lm_torch(itr=itr, kl=kl)
model = model_complete(lm_torch, mnist_model)


# Attacking function
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon, num_steps=1):
    correct = 0
    adv_examples = []
    # Loop over all examples in test set
    for data, target in tqdm.tqdm(test_loader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Make data suitable
        data = data.reshape(1, 28, 28)
        # target = np.array([1*(idx == target) for idx in range(10)])

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model.forward(data)

        # Get the best index in output
        init_pred = torch.argmax(output)

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        print(target)

        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # If requested, reconstruct the input iterating forward-backward dynamics
        if num_steps != 1:
            perturbed_data = gen_image(
                None, FLAGS, model.mnist_model, perturbed_data, num_steps=num_steps, sample=False)[0]

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

            # Special case for saving 0 epsilon examples
            if (epsilon == 0):  # and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 1000:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
          correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def test_adv(model, dataloader, epsilons=list([0, .05, .1, .15, .2, .25, .3]), steps=1):
    accuracies = list()
    examples = list()

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps, num_steps=steps)
        accuracies.append(acc)
        # examples.append(ex)

    return accuracies, examples


# Implement test function to get results related at more expsilons
acc, exs = test_adv(model=model, dataloader=test_loader, steps=1)


# Test it and save variables
pickle.dump(acc, open(f'vars/accuracies_ro_adv{itr}{kl_str}.pkl', 'wb'))

# Preprocess the dataset, for iterations tuning
df_final = [(exs[0][i][2], exs[0][i][0]) for i in range(1000)]
dataloader = torch.utils.data.DataLoader(df_final)

# Set the right number of denoising iterations
accs = list()
for steps in [10*i for i in range(1, 11)]:
    acc = test(model=model, device=device, test_loader=dataloader,
               epsilon=.3, num_steps=steps)[0]
    accs.append(acc)
    print('The steps are ', steps, ' and the accuracy is ', acc)


# Plot final graph
pickle.dump(kl8, open('vars/accuracies_ro_adv8.pkl', 'wb'))
kl8 = acc
sns.set_style("darkgrid")
plt.xlabel('Epsilon value of noise injection process')
plt.ylabel('Accuracy of final read out')
plt.title('accuracy read-out values per growing adversarial attacking')

plt.ylim(0, 1)
sns.lineplot(epsilons, nokl2, color='lightblue')
sns.lineplot(epsilons, kl2, color='darkblue')
sns.lineplot(epsilons, nokl8, color='red')
sns.lineplot(epsilons, kl8, color='darkred')
# sns.lineplot(std_values, acc_48_n, color='purple')
# sns.lineplot(std_values, acc_48, color='brown')

plt.legend(['CondRes 256 NOKL', 'CondRes 256 KL',
           'CondRes 1024 NOKL', 'CondRes 1024 KL'])


# %%
