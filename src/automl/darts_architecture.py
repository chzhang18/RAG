import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, lr_a, momentum, weight_decay, weight_decay_a):
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay
        self.model = model
        self.virtual_model = deepcopy(self.model)

        params = []
        for p in self.model.arch_parameters().values():
            params.append(p)

        self.optimizer = torch.optim.Adam(params=params,
                                          lr=lr_a, betas=(0.5, 0.999),
                                          weight_decay=weight_decay_a)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model.loss(input_valid, target_valid)
        loss.backward()

    def virtual_step(self, input_train, target_train, eta, network_optimizer):
        loss = self.model.loss(input_train, target_train)
        gradients = torch.autograd.grad(loss, self.model.parameters())

        with torch.no_grad():
            for p, vp, g in zip(self.model.parameters(), self.virtual_model.parameters(), gradients):
                m = network_optimizer.state[p].get('momentum_buffer', 0) * self.network_momentum
                vp.copy_(p - eta * (m+g+self.network_weight_decay*p))

            for a, va in zip(self.model.arch_parameters(), self.virtual_model.arch_parameters()):
                va.copy_(a)

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        self.virtual_step(input_train, target_train, eta, network_optimizer)
        loss = self.virtual_model.loss(input_valid, target_valid)

        # compute gradient
        v_a = tuple(self.virtual_model.arch_parameters())
        v_p = tuple(self.virtual_model.parameters())
        v_grad = torch.autograd.grad(loss, v_a+v_p)
        d_a = v_grad[:len(v_a)]
        d_p = v_grad[len(v_a):]

        hessian = self._hessian_vector_product(d_p, input_train, target_train)

        with torch.no_grad():
            for a, d, h in zip(self.model.arch_parameters(), d_a, hessian):
                a.grad = d - eta * h

    def _hessian_vector_product(self, dw, input_train, target_train, r=1e-2):
        eps = r / torch.cat([w.view(-1) for w in dw]).norm()

        # w+ = w + eps * dw
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p += eps * d
        loss = self.model.loss(input_train, target_train)
        da_pos = torch.autograd.grad(loss, self.model.arch_parameters())

        # w- = w - eps * dw
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p -= 2 * eps * d
        loss = self.model.loss(input_train, target_train)
        da_neg = torch.autograd.grad(loss, self.model.arch_parameters())

        # recover w
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p += eps * d

        hessian = [(p-n) / 2.0 * eps for p, n in zip(da_pos, da_neg)]

        return hessian
