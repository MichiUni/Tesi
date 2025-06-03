#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:41:17 2021
@author: laurent
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class VIModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internalLosses = []
        self.lossScaleFactor = 1

    def addLoss(self, func):
        self._internalLosses.append(func)

    def evalLosses(self):
        t_loss = 0
        for l in self._internalLosses:
            t_loss += l(self)
        return t_loss

    def evalAllLosses(self):
        t_loss = self.evalLosses() * self.lossScaleFactor
        for m in self.children():
            if isinstance(m, VIModule):
                t_loss += m.evalAllLosses() * self.lossScaleFactor
        return t_loss

class L2RegularizedLinear(VIModule, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, wPriorSigma=1., bPriorSigma=1., bias_init_cst=0.0):
        super().__init__(in_features, out_features, bias=bias)
        if bias:
            self.bias.data.fill_(bias_init_cst)
        self.addLoss(lambda s: 0.5 * s.weight.pow(2).sum() / wPriorSigma**2)
        if bias:
            self.addLoss(lambda s: 0.5 * s.bias.pow(2).sum() / bPriorSigma**2)

class MeanFieldGaussianFeedForward(VIModule):
    def __init__(self, in_features, out_features, bias=True, groups=1, weightPriorMean=0, weightPriorSigma=1.,
                 biasPriorMean=0, biasPriorSigma=1., initMeanZero=False, initBiasMeanZero=False, initPriorSigmaScale=0.01):
        super().__init__()
        self.samples = {'weights': None, 'bias': None, 'wNoiseState': None, 'bNoiseState': None}
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weights_mean = Parameter((0. if initMeanZero else 1.) * (torch.rand(out_features, int(in_features / groups)) - 0.5))
        self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale * weightPriorSigma * torch.ones(out_features, int(in_features / groups))))
        self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features / groups)),
                                         torch.ones(out_features, int(in_features / groups)))

        self.addLoss(lambda s: 0.5 * s.getSampledWeights().pow(2).sum() / weightPriorSigma**2)
        self.addLoss(lambda s: -s.out_features / 2 * np.log(2 * np.pi) - 0.5 * s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())

        if self.has_bias:
            self.bias_mean = Parameter((0. if initBiasMeanZero else 1.) * (torch.rand(out_features) - 0.5))
            self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale * biasPriorSigma * torch.ones(out_features)))
            self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
            self.addLoss(lambda s: 0.5 * s.getSampledBias().pow(2).sum() / biasPriorSigma**2)
            self.addLoss(lambda s: -s.out_features / 2 * np.log(2 * np.pi) - 0.5 * s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())

    def sampleTransform(self, stochastic=True):
        self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
        self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma) * self.samples['wNoiseState'] if stochastic else 0)
        if self.has_bias:
            self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma) * self.samples['bNoiseState'] if stochastic else 0)

    def getSampledWeights(self):
        return self.samples['weights']

    def getSampledBias(self):
        return self.samples['bias']

    def forward(self, x, stochastic=True):
        self.sampleTransform(stochastic=stochastic)
        return nn.functional.linear(x, self.samples['weights'], bias=self.samples['bias'] if self.has_bias else None)

class BayesianEmulator(VIModule):
    def __init__(self, weight_sigma=1., bias_sigma=1.):
        super().__init__()
        self.hidden = MeanFieldGaussianFeedForward(
            in_features=2,
            out_features=20,
            weightPriorSigma=weight_sigma,
            biasPriorSigma=bias_sigma
        )
        self.output = MeanFieldGaussianFeedForward(
            in_features=20,
            out_features=3,
            weightPriorSigma=weight_sigma,
            biasPriorSigma=bias_sigma
        )

    def forward(self, x, stochastic=True):
        h = torch.relu(self.hidden(x, stochastic=stochastic))
        out = self.output(h, stochastic=stochastic)
        return out

class StandardEmulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 20)
        self.output = nn.Linear(20, 3)

    def forward(self, x):
        h = torch.relu(self.hidden(x))
        return self.output(h)