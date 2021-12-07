import torch
import torch.nn.functional as F
import tqdm

from flags import get_flags, gen_image


# FLAGS = get_flags()

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon, FLAGS, num_steps=1):
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
        # print(target)

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


def test_adv(model, dataloader, FLAGS, epsilons=list([0, .05, .1, .15, .2, .25, .3]), steps=1):
    accuracies = list()
    examples = list()

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps, FLAGS, num_steps=steps)
        accuracies.append(acc)
        # examples.append(ex)

    return accuracies, examples
