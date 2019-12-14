from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import os
os.chdir('../')
import torch_vr
os.chdir('../torch_sample')
import torch_sample
os.chdir('../torch_vr/examples')


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


def sample(X, y, model, nlp_func, start, step_size, batch_size, accelerated=False, gam=1.0, var_reduce=None, iterations=100):

    # loads the starting point
    model.load_state_dict(start)

    # initializes the optimizer
    if accelerated:
        sampler = torch_sample.unadjusted.ULD(model.parameters(), t=step_size, gam=gam)
    else:
        sampler = torch_sample.unadjusted.LD(model.parameters(), t=step_size)

    # initializes a variance reducer if any requested
    if var_reduce in ['SAG', 'SAGA']:
        var_reducer = torch_vr.ReduceVar(
            model.parameters(), X.shape[0], model.layers, method=var_reduce)

    # initializes a list containing the losses
    nlps = list()

    for _ in tqdm(range(iterations)):
        # resets the gradients to 0
        sampler.zero_grad()

        # samples the mini-batch indices uniformly without replacement
        indices = np.random.choice(X.shape[0], batch_size, replace=False)

        # computes the loss
        nlp = nlp_func(model(X[indices]), y[indices])
        nlp.backward()

        # constructs the gradient estimate
        if var_reduce is None:
            for p in model.parameters():
                p.grad *= X.shape[0]/batch_size
        else:
            var_reducer.reduce_variance(model.outputs_grad, model.inputs, indices)

        # records the true loss at the current iterate
        with torch.no_grad():
            nlps.append(nlp_func(model(X), y))

        # takes an optimization step
        sampler.transition()

    return nlps
