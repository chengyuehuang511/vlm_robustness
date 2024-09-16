import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import copy
import math
import logging
from typing import List, Dict, Optional
import torch.nn.functional as F


def orthogonal_component(a, b, strength=1.0, return_projection=False):
    assert strength >= 0.0 and strength <= 1.0, "Invalid strength value: {}".format(strength)

    # Ensure b is not a zero matrix
    if torch.allclose(b, torch.zeros_like(b)):
        raise ValueError("Matrix b must not be the zero matrix.")
    
    # Calculate the projection of a onto b
    b_norm_squared = torch.sum(b * b, dim=-1, keepdim=True)
    projection = torch.sum(a * b, dim=-1, keepdim=True) / b_norm_squared * b
    
    # Subtract the projection from a to get the orthogonal component
    orthogonal = a - strength * projection
    
    if return_projection:
        return projection
    return orthogonal


class SPCG(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, exclude_set={}, use_lora=False):
        self.exclude_set = exclude_set
        self.use_lora = use_lora
        
        print("Weight decay is the orthogonal component strength!")

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SPCG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SPCG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            hyper_param = []
            max_exp_avg_sqs = []
            state_steps = []
            condition_buffer = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['hyper'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                       

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    hyper_param.append(state['hyper'])
                    # initalize condition_buffer
                    condition_buffer.append(torch.tensor(0,dtype=torch.float).to(p.device))

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(group,
                   exp_avgs,
                   exp_avg_sqs,
                   hyper_param,
                   max_exp_avg_sqs,
                   condition_buffer,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss

    def adam(self, 
            group: Dict[str, List[torch.Tensor]],
            exp_avgs: List[torch.Tensor],
            exp_avg_sqs: List[torch.Tensor],
            hyper_param: Dict[str, float],
            max_exp_avg_sqs: Optional[List[torch.Tensor]],
            condition_buffer: List[torch.Tensor],
            state_steps: List[int], 
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float):
            
        def compute_denominator(exp_avg_sq, bias_correction2, max_exp_avg_sq=None):
            if amsgrad:
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                return (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                return (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        def update_parameter(param, grad, exp_avg, exp_avg_sq, step, pre=None):
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            """projecting conflicting gradients"""
            condition = (param if pre is None else param - pre)
            # condition_buffer[i] += torch.sum(grad * condition)
            # if condition_buffer[i] < 0.0:
            dot = torch.sum(grad * condition)
            grad_norm = torch.norm(grad)
            condition_norm = torch.norm(condition)

            lamb = 0
            if dot < 0.0:
                lamb = -weight_decay * (dot / (condition_norm ** 2))
                grad += lamb * condition
            print(lamb)
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = compute_denominator(exp_avg_sq, bias_correction2, max_exp_avg_sqs[i] if amsgrad else None)
            step_size = lr / bias_correction1
            
            d_p = step_size * exp_avg / denom
            param.copy_(param - d_p)
        
        for i, param in enumerate(group['params']):
            if param.grad is None: 
                continue
            
            grad = param.grad
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            if self.use_lora:
                update_parameter(param, grad, exp_avg, exp_avg_sq, step)
            else:
                pre = group['pre'][i]
                update_parameter(param, grad, exp_avg, exp_avg_sq, step, pre)