from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch
import os
os.chdir('../')
import torch_vr
os.chdir('../torch_sample')
import torch_sample

def optimize(inputs_train, targets_train, inputs_test, targets_test, model,
             loss_func, accuracy, start, step_size, epochs=100, accelerated=False,
             momentum=0.9, stochastic=False, batch_size=128, var_reduce=None, ranks=None):

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
            model.parameters(), inputs_train.shape[0], model.layers, method=var_reduce, ranks=ranks)
    # initializes a list containing the accuracies
    losses_train = list()
    losses_test = list()
    accuracies_train = list()
    accuracies_test = list()
    # optimizes
    for _ in tqdm(range(epochs)):
        for _ in range(int(inputs_train.shape[0]/batch_size)):
            # computes the loss
            if stochastic:
                indices = np.random.choice(inputs_train.shape[0], batch_size, replace=False)
                if var_reduce in ['SAG', 'SAGA']:
                    loss = loss_func(model(inputs_train[indices], cache=True), targets_train[indices])
                else:
                    loss = loss_func(model(inputs_train[indices]), targets_train[indices])
            else:
                loss = loss_func(model(inputs_train), targets_train)
            # computes the gradient
            loss.backward()
            # constructs the gradient estimate
            if stochastic:
                if var_reduce not in ['SAG', 'SAGA']:
                    with torch.no_grad():
                        for p in model.parameters():
                            p.grad *= inputs_train.shape[0]/batch_size
                else:
                    var_reducer.reduce_variance(model.outputs_grad, model.inputs, indices)
            # takes an optimization step
            opt.step()
            # resets the gradients to 0
            opt.zero_grad()
        # records the accuracy and loss at every epoch
        with torch.no_grad():
            outputs_train = model(inputs_train)
            accuracies_train.append(accuracy(outputs_train, targets_train))
            losses_train.append(loss_func(outputs_train, targets_train))
            outputs_test = model(inputs_test)
            accuracies_test.append(accuracy(outputs_test, targets_test))
            losses_test.append(loss_func(outputs_test, targets_test))

    return (accuracies_train, losses_train, accuracies_test, losses_test)

def sample(inputs_train, targets_train, inputs_test, targets_test, model,
           nlp_func, accuracy, collect_samples, start, step_size, epochs=100, accelerated=False,
           momentum=0.9, stochastic=False, batch_size=128, var_reduce=None, ranks=None):

    # loads the starting point
    model.load_state_dict(start)
    # initializes the sampler
    if accelerated:
        t = -step_size * torch.log(torch.tensor(momentum))/(1 - momentum)
        gam = -torch.log(torch.tensor(momentum))/t
        sampler = torch_sample.unadjusted.ULD(model.parameters(), t=t, gam=gam)
    else:
        sampler = torch_sample.unadjusted.LD(model.parameters(), t=step_size)
    # makes sure the gradients are cleared
    sampler.zero_grad()
    # initializes a variance reducer if any requested
    if var_reduce in ['SAG', 'SAGA'] and stochastic:
        var_reducer = torch_vr.ReduceVar(
            model.parameters(), inputs_train.shape[0], model.layers, method=var_reduce, ranks=ranks)
    # initializes a list containing the accuracies
    # and a list containing the samples if requested
    accuracies_train = list()
    accuracies_test = list()
    if collect_samples:
        samples = list()
        for p in model.parameters():
            samples.append(torch.zeros([epochs] + list(p.shape)))
    # samples
    for epoch in tqdm(range(epochs)):
        for _ in range(int(inputs_train.shape[0]/batch_size)):

            # computes the negative log posterior
            if stochastic:
                indices = np.random.choice(inputs_train.shape[0], batch_size, replace=False)
                if var_reduce in ['SAG', 'SAGA']:
                    nlp = nlp_func(model(inputs_train[indices], cache=True), targets_train[indices])
                else:
                    nlp = nlp_func(model(inputs_train[indices]), targets_train[indices])
            else:
                nlp = nlp_func(model(inputs_train), targets_train)
            # computes the gradient
            nlp.backward()
            # constructs the gradient estimate
            if stochastic:
                if var_reduce not in ['SAG', 'SAGA']:
                    with torch.no_grad():
                        for p in model.parameters():
                            p.grad *= inputs_train.shape[0]/batch_size
                else:
                    var_reducer.reduce_variance(model.outputs_grad, model.inputs, indices)
            # transitions to the next state
            sampler.transition()
            # resets the gradients to 0
            sampler.zero_grad()
        # records the accuracy every epoch
        with torch.no_grad():
            accuracies_test.append(accuracy(inputs_test, targets_test))
            accuracies_train.append(accuracy(inputs_train, targets_train))
        # records the generated sample at every epoch
        if collect_samples:
            for i, p in enumerate(model.parameters()):
                samples[i][epoch].copy_(p.detach())

    if collect_samples:
        return (accuracies_train, accuracies_test, samples)
    else:
        return (accuracies_train, accuracies_test)
