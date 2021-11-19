import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from models_utils import BNN
from pmnist_robustness_data_utils import TaskDataSet

# Note: Change test dataset path and model checkpoints path before running the script
# Reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def plot_examples(examples, epsilons):
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))
    ])

    common_dload_args = dict(
        batch_size=1
        # num_workers = args.num_workers,
        # pin_memory  = args.pin_memory
    )

    accuracies = []
    examples = []
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for meta in ['0.00', '0.70', '1.35']:
        for task in range(1, 7):
            test_dataset = TaskDataSet(
                f'/content/bnn-cf-vs-robust/data/input/pmnist_robustness/task-0{str(task)}/original/test',
                transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, **common_dload_args)
            ckpt = torch.load(
                f'/content/bnn-cf-vs-robust/data/output/pmnist_robustness_[2048x2048]_[meta={meta}]/models/task-0{str(task)}.pt')
            model = BNN(**ckpt['model_args'])
            model.load_state_dict(ckpt['model_states'])
            model.eval()
            model.cuda()
            # Run test for each epsilon
            exa_i, acc_i = [], []
            for eps in epsilons:
                acc, ex = test(model, 'cuda:0', test_loader, eps)
                acc_i.append(acc)
                exa_i.append(ex)
            accuracies.append(acc_i)
            examples.append(exa_i)
    plt.close('all')

    plt.figure(figsize=(15, 15))
    plt.plot([], [], ' ', label='meta, train_pha')
    plt.plot(epsilons, accuracies[0], "r.-", label='0.00, task-01')
    plt.plot(epsilons, accuracies[1], "rv-", label='0.00, task-02')
    plt.plot(epsilons, accuracies[2], "r^-", label='0.00, task-03')
    plt.plot(epsilons, accuracies[3], "rs-", label='0.00, task-04')
    plt.plot(epsilons, accuracies[4], "rh-", label='0.00, task-05')
    plt.plot(epsilons, accuracies[5], "rd-", label='0.00, task-06')
    plt.plot(epsilons, accuracies[6], "b.-", label='0.70, task-01')
    plt.plot(epsilons, accuracies[7], "bv-", label='0.70, task-02')
    plt.plot(epsilons, accuracies[8], "b^-", label='0.70, task-03')
    plt.plot(epsilons, accuracies[9], "bs-", label='0.70, task-04')
    plt.plot(epsilons, accuracies[10], "bh-", label='0700, task-05')
    plt.plot(epsilons, accuracies[11], "bd-", label='0.70, task-06')
    plt.plot(epsilons, accuracies[12], "g.-", label='1.35, task-01')
    plt.plot(epsilons, accuracies[13], "gv-", label='1.35, task-02')
    plt.plot(epsilons, accuracies[14], "g^-", label='1.35, task-03')
    plt.plot(epsilons, accuracies[15], "gs-", label='1.35, task-04')
    plt.plot(epsilons, accuracies[16], "gh-", label='1.35, task-05')
    plt.plot(epsilons, accuracies[17], "gd-", label='1.35, task-06')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.show()
