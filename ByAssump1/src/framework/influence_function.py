#! /usr/bin/env python3
import random
from random import sample
import torch
from torch.autograd import grad
from torch import nn
#from pytorch_influence_functions.utils import display_progress
random.seed(1219)
torch.manual_seed(1219)
torch.cuda.manual_seed_all(1219)
torch.backends.cudnn.deterministic = True

def s_test(ins_data, model, z_loader, criterion, damp=0.01, scale=25.0,recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(ins_data, model, criterion)
    h_estimate = v.copy()
    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    #for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
    '''
    for k, idx in enumerate(train_sample) :
        ins_data = z_loader.dataset[idx]
        if torch.cuda.is_available():
            for i in range(1,len(ins_data)):
                ins_data[i] = ins_data[i].cuda()
        label = ins_data[0]
        if label != 0 :
            label = 1
        args = ins_data[1:]
        logits = model(*args)
        loss = criterion(logits, torch.LongTensor([label]).cuda())
        hv = hvp(loss,  [p for p in model.parameters() if p.requires_grad][3:], h_estimate)           # Recursively caclulate h_estimate
        # Recursively caclulate h_estimate
        h_estimate = [
            _v + (1 - damp) * _h_e - _hv.detach() / scale
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
    '''
    def to_flat_tensor(tensors) :
        return torch.cat([t.flatten() for t in tensors])
    def norm(tensors) :
        return torch.norm(to_flat_tensor(tensors))
    def normalize(tensors) :
        norm = torch.norm(to_flat_tensor(tensors))
        return [t/norm for t in tensors]
    for idx in range(5) :
        for k, data in enumerate(z_loader) :
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            args = data[1:]
            logits = model(*args)
            loss = criterion(logits, label)
            hv = hvp(loss, [p for p in model.parameters() if p.requires_grad], h_estimate)           # Recursively caclulate h_estimate
            h_estimate = [
            _v + (1 - damp) * _h_e - _hv.detach() / scale
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            #print(norm(h_estimate))
    h_estimate = [h_e / scale for h_e in h_estimate]
    '''
    grad_v = grad_z(ins_data, model, criterion)
    tmp_influence = sum(
                [
                    ###################################
                    # TODO: verify if computation really needs to be done
                    # on the CPU or if GPU would work, too
                    ###################################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_v, h_estimate)
                    ###################################
                    # Originally with [i] because each grad_z contained
                    # a list of tensors as long as e_s_test list
                    # There is one grad_z per training data sample
                    ###################################
                    ])
    print(tmp_influence)
    '''
    return h_estimate

def grad_z(ins_data, model, criterion):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    if torch.cuda.is_available():
        for i in range(1,len(ins_data)):
            ins_data[i] = ins_data[i].cuda()
    label = ins_data[0]
    args = ins_data[1:]
    logits = model(*args)
    loss = criterion(logits, torch.LongTensor([label]).cuda())


    # Compute sum of gradients from model parameters to loss
    return list(grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads
