import torch
from utils.arch import apply_neuron_mask


def collect_mask_grads(model, head_mask, neuron_mask, dataloader) :
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)

    handles = apply_neuron_mask(model, neuron_mask)

    model.eval()
    head_grads = list()
    neuron_grads = list()

    for batch in dataloader :
        for k,v in batch.items() :
            batch[k] = v.to("cuda", non_blocking = True)

        output = model(head_mask = head_mask, **batch)
        loss = output.loss
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        neuron_grads.append(neuron_mask.grad.detach())
        neuron_mask.grad = None

        for handle in handles:
            handle.remove()

        head_mask.requires_grad_(False)
        neuron_mask.requires_grad_(False)

        head_grads = torch.stack(head_grads, dim = 0)
        neuron_grads = torch.stack(neuron_grads, dim = 0)
        return head_grads, neuron_grads


@torch.no_grad()
def compute_covariance_matrix(grads) :
    grads =  grads.view(grads.size(0), -1)
    mean_grad = torch.mean(grads, dim = 0)
    grads = grads - mean_grad
    cov_grads = torch.matmul(grads.t(), grads) / grads.size(0)
    return cov_grads


@torch.no_grad()
def compute_importance_scores(grads) :
    batch, l, h = grads.size(0), grads.size(1), grads.size(2)
    grads = compute_covariance_matrix(grads)
    eigenvalues = torch.linalg.eigvalsh(grads)
    eigenvalues = eigenvalues ** 2
    total_var = eigenvalues.sum()

    importance_score = eigenvalues / total_var
    return importance_score.view(l, h)


@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info
