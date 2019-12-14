from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import os
os.chdir('../')
import torch_vr

def optimize(inputs_train, targets_train, inputs_test, targets_test, model,
             loss_func, accuracy, start, step_size, iterations=100, accelerated=False,
             momentum=0.9, stochastic=False, batch_size=128, var_reduce=None):

    # loads the starting point
    model.load_state_dict(start)

    # initializes the optimizer
    if accelerated:
        opt = torch.optim.SGD(model.parameters(), lr=step_size, momentum=momentum, nesterov=True)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=step_size)

    # makes sure the gradients are cleared
    opt.zero_grad()

    # initializes a variance reducer if any requested
    if var_reduce in ['SAG', 'SAGA'] and stochastic:
        var_reducer = torch_vr.ReduceVar(
            model.parameters(), inputs_train.shape[0], model.layers, method=var_reduce)

    # initializes a list containing the accuracies
    accuracies_train = list()
    accuracies_test = list()

    for _ in tqdm(range(iterations)):

        # computes the loss
        if stochastic:
            # samples the mini-batch indices uniformly without replacement
            indices = np.random.choice(inputs_train.shape[0], batch_size, replace=False)

            # computes the loss
            loss = loss_func(model(inputs_train[indices]), targets_train[indices])
        else:
            loss = loss_func(model(inputs_train), targets_train)

        # computes the gradient
        loss.backward()

        # constructs the gradient estimate
        if stochastic:
            if var_reduce not in ['SAG', 'SAGA']:
                for p in model.parameters():
                    p.grad *= inputs_train.shape[0]/batch_size
            else:
                var_reducer.reduce_variance(model.outputs_grad, model.inputs, indices)

        # records the accuracy every 100 iterations
        with torch.no_grad():
            accuracies_test.append(accuracy(inputs_train, targets_test))
            accuracies_train.append(accuracy(inputs_test, targets_test))

        # takes an optimization step
        opt.step()

        # resets the gradients to 0
        opt.zero_grad()

    return (accuracies_train, accuracies_test)
