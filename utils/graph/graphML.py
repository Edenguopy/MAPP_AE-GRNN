# 2018/11/01~2018/07/12
# Fernando Gama, fgama@seas.upenn.edu.
# Luana Ruiz, rubruiz@seas.upenn.edu.
# Kate Tolstaya, eig@seas.upenn.edu
"""
图信号处理和图机器学习的模块，
主要用于构建图神经网络模型、实现图滤波操作及相关的深度学习组件
"""
"""
graphML.py Module for basic GSP and graph machine learning functions.

Functionals

LSIGF: Applies a linear shift-invariant graph filter
spectralGF: Applies a linear shift-invariant graph filter in spectral form
NVGF: Applies a node-variant graph filter
EVGF: Applies an edge-variant graph filter
learnAttentionGSO: Computes the GSO following the attention mechanism
graphAttention: Applies a graph attention layer

Filtering Layers (nn.Module)

GraphFilter: Creates a graph convolutional layer using LSI graph filters
SpectralGF: Creates a graph convolutional layer using LSI graph filters in
    spectral form
NodeVariantGF: Creates a graph filtering layer using node-variant graph filters
EdgeVariantGF: Creates a graph filtering layer using edge-variant graph filters
GraphAttentional: Creates a layer using graph attention mechanisms

Activation Functions - Nonlinearities (nn.Module)

MaxLocalActivation: Creates a localized max activation function layer
MedianLocalActivation: Creates a localized median activation function layer
NoActivation: Creates a layer for no activation function

Summarizing Functions - Pooling (nn.Module)

NoPool: No summarizing function.
MaxPoolLocal: Max-summarizing function
"""

import math
import numpy as np
import torch
import torch.nn as nn

import utils.graph.graphTools as gt

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

# WARNING: Only scalar bias.

#############################################################################
#                                                                           #
#                           FUNCTIONALS                                     #
#                                                                           #
#############################################################################

def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def spectralGF(h, V, VH, x, b=None):
    """
    spectralGF(filter_coeff, eigenbasis, eigenbasis_hermitian, input, bias=None)
        Computes the output of a linear shift-invariant graph filter in spectral
        form applying filter_coefficients on the graph fourier transform of the
        input .

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, N the number of nodes, S_{e} in R^{N x N}
    the GSO for edge feature e with S_{e} = V_{e} Lambda_{e} V_{e}^{H} as
    eigendecomposition, x in R^{G x N} the input data where x_{g} in R^{N} is
    the graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the LSI-GF in spectral form is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{g=1}^{G}
                        V_{e} diag(h_{f,g,e}) V_{e}^{H} x_{g}
                + b_{f}
    for f = 1, ..., F, with h_{f,g,e} in R^{N} the filter coefficients for
    output feature f, input feature g and edge feature e.

    Inputs:
        filter_coeff (torch.tensor): array of filter coefficients; shape:
            output_features x edge_features x input_features x number_nodes
        eigenbasis (torch.tensor): eigenbasis of the graph shift operator;shape:
            edge_features x number_nodes x number_nodes
        eigenbasis_hermitian (torch.tensor): hermitian of the eigenbasis; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes

    Obs.: While we consider most GSOs to be normal (so that the eigenbasis is
    an orthonormal basis), this function would also work if V^{-1} is used as
    input instead of V^{H}
    """
    # The decision to input both V and V_H is to avoid any time spent in
    # permuting/inverting the matrix. Because this depends on the graph and not
    # the data, it can be done faster if we just input it.

    # h is output_features x edge_weights x input_features x number_nodes
    # V is edge_weighs x number_nodes x number_nodes
    # VH is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    G = h.shape[2]
    N = h.shape[3]
    assert V.shape[0] == VH.shape[0] == E
    assert V.shape[1] == VH.shape[1] == V.shape[2] == VH.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x G x N
    # V in E x N x N
    # VH in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # We will do proper matrix multiplication in this case (algebraic
    # multiplication using column vectors instead of CS notation using row
    # vectors).
    # We will multiply separate VH with x, and V with diag(h).
    # First, to multiply VH with x, we need to add one dimension for each one
    # of them (dimension E for x and dimension B for VH)
    x = x.reshape([B, 1, G, N]).permute(0, 1, 3, 2) # B x 1 x N x G
    VH = VH.reshape([1, E, N, N]) # 1 x E x N x N
    # Now we multiply. Note that we also permute to make it B x E x G x N
    # instead of B x E x N x G because we want to multiply for a specific e and
    # g, there we do not want to sum (yet) over G.
    VHx = torch.matmul(VH, x).permute(0, 1, 3, 2) # B x E x G x N

    # Now we want to multiply V * diag(h), both are matrices. So first, we
    # add the necessary dimensions (B and G for V and an extra N for h to make
    # it a matrix from a vector)
    V = V.reshape([1, E, 1, N, N]) # 1 x E x 1 x N x N
    # We note that multiplying by a diagonal matrix to the right is equivalent
    # to an elementwise multiplication in which each column is multiplied by
    # a different number, so we will do this to make it faster (elementwise
    # multiplication is faster than matrix multiplication). We need to repeat
    # the vector we have columnwise.
    diagh = h.reshape([F, E, G, 1, N]).repeat(1, 1, 1, N, 1) # F x E x G x N x N
    # And now we do elementwise multiplication
    Vdiagh = V * diagh # F x E x G x N x N
    # Finally, we make the multiplication of these two matrices. First, we add
    # the corresponding dimensions
    Vdiagh = Vdiagh.reshape([1, F, E, G, N, N]) # 1 x F x E x G x N x N
    VHx = VHx.reshape([B, 1, E, G, N, 1]) # B x 1 x E x G x N x 1
    # And do matrix multiplication to get all the corresponding B,F,E,G vectors
    VdiaghVHx = torch.matmul(Vdiagh, VHx) # B x F x E x G x N x 1
    # Get rid of the last dimension which we do not need anymore
    y = VdiaghVHx.squeeze(5) # B x F x E x G x N
    # Sum over G
    y = torch.sum(y, dim = 3) # B x F x E x N
    # Sum over E
    y = torch.sum(y, dim = 2) # B x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def NVGF(h, S, x, b=None):
    """
    NVGF(filter_taps, GSO, input, bias=None) Computes the output of a
    node-variant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of shifts, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f. Denote as h_{k}^{efg} in R^{N} the vector with the N
    filter taps corresponding to the efg filter for shift k.

    Then, the NV-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        diag(h_{k}^{efg}) S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
                x number_nodes
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # h is output_features x edge_weights x filter_taps x input_features
    #                                                             x number_nodes
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    N = h.shape[4]
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x K x G x N
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    xr = x.reshape([B, 1, G, N])
    Sr = S.reshape([1, E, N, N])
    z = xr.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        xr = torch.matmul(xr, Sr) # B x E x G x N
        xS = xr.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # This multiplication with filter taps is ``element wise'' on N since for
    # each node we have a different element
    # First, add the extra dimension (F for z, and B for h)
    z = z.reshape([B, 1, E, K, G, N])
    h = h.reshape([1, F, E, K, G, N])
    # Now let's do elementwise multiplication
    zh = z * h
    # And sum over the dimensions E, K, G to get B x F x N
    y = torch.sum(zh, dim = 4) # Sum over G
    y = torch.sum(y, dim = 3) # Sum over K
    y = torch.sum(y, dim = 2) # Sum over E
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

def EVGF(S, x, b=None):
    """
    EVGF(filter_matrices, input, bias=None) Computes the output of an
    edge-variant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of shifts, N the number of
    nodes, Phi_{efg} in R^{N x N} the filter matrix for edge feature e, output
    feature f and input feature g (recall that Phi_{efg}^{k} has the same
    sparsity pattern as the graph, except for Phi_{efg}^{0} which is expected to
    be a diagonal matrix), x in R^{G x N} the input data where x_{g} in R^{N} is
    the graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the EV-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        Phi_{efg}^{k:0} x_{g}
                + b_{f}
    for f = 1, ..., F, with Phi_{efg}^{k:0} = Phi_{efg}^{k} Phi_{efg}^{k-1} ...
    Phi_{efg}^{0}.

    Inputs:
        filter_matrices (torch.tensor): array of filter matrices; shape:
            output_features x edge_features x filter_taps x input_features
                x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # We just need to multiply by the filter_matrix recursively, and then
    # add for all E, G, and K features.

    # S is output_features x edge_features x filter_taps x input_features
    #   x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = S.shape[0]
    E = S.shape[1]
    K = S.shape[2]
    G = S.shape[3]
    N = S.shape[4]
    assert S.shape[5] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # S in F x E x K x G x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # We will be doing matrix multiplications in the algebraic way, trying to
    # multiply the N x N matrix corresponding to the appropriate e, f, k and g
    # dimensions, with the respective x vector (N x 1 column vector)
    # For this, we first add the corresponding dimensions (for x we add
    # dimensions F, E and the last dimension for column vector)
    x = x.reshape([B, 1, 1, G, N, 1])
    # When we do index_select along dimension K we get rid of this dimension
    Sk = torch.index_select(S, 2, torch.tensor(0).to(S.device)).squeeze(2)
    # Sk in F x E x G x N x N
    # And we add one further dimension for the batch size B
    Sk = Sk.unsqueeze(0) # 1 x F x E x G x N x N
    # Matrix multiplication
    x = torch.matmul(Sk, x) # B x F x E x G x N x 1
    # And we collect this for every k in a vector z, along the K dimension
    z = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
    # Now we do all the matrix multiplication
    for k in range(1,K):
        # Extract the following k
        Sk = torch.index_select(S, 2, torch.tensor(k).to(S.device)).squeeze(2)
        # Sk in F x E x G x N x N
        # Give space for the batch dimension B
        Sk = Sk.unsqueeze(0) # 1 x F x E x G x N x N
        # Multiply with the previously cumulative Sk * x
        x = torch.matmul(Sk, x) # B x F x E x G x N x 1
        # Get rid of the last dimension (of a column vector)
        Sx = x.reshape([B, F, E, 1, G, N, 1]).squeeze(6) # B x F x E x 1 x G x N
        # Add to the z
        z = torch.cat((z, Sx), dim = 2) # B x F x E x k x G x N
    # Sum over G
    z = torch.sum(z, dim = 4)
    # Sum over K
    z = torch.sum(z, dim = 3)
    # Sum over E
    y = torch.sum(z, dim = 2)
    if b is not None:
        y = y + b
    return y  

def jARMA(psi, varphi, phi, S, x, b=None, tMax = 5):
    """
    jARMA(inverse_taps, direct_taps, filter_taps, GSO, input, bias = None,
        tMax = 5) Computes the output of an ARMA filter using Jacobi
        iterations.
        
    The output of an ARMA computed by means of tMax Jacobi iterations is given
    as follows
        y^{f} = \sum_{e=1}^{E} \sum_{g=1}^{G}
                    \sum_{p=0}^{P-1}
                        H_{p}^{1}(S) (\bar{S}_{p}^{fge})^{-1} x
                        + H_{p}^{2}(S) x
                    + H^{3}(S) x
    where E is the total number of edge features, G is the total number of input
    features, and P is the order of the denominator polynomial. The filters are
        H_{p}^{1}(S) 
            = \sum_{tau=0}^{t} 
                (-1)^{tau} varphi_{p}^{fge} (\barS_{p}^{-1} \tilde{S})^{tau}
        H_{p}^{2}(S)
            = (-1)^{t+1} ((\bar{S}_{p}^{fge})^{-1} \tilde{S})^{t+1}
        H^{2}(S) = \sum_{k=0}^{K-1} phi_{k}^{fge} S^{k}
    where varphi_{p}^{fge} are the direct filter taps of the rational one ARMA 
    filter, phi_{k}^{fge} are the filter taps of the residue LSIGF filter, and
    the GSOs used derive from GSO S and are
        \bar{S}_{p}^{fge} = Diag(S) - psi_{p}^{fge} I_{N}
        \tilde{S} = DiagOff(S)
    with psi_{p}^{fge} the inverse filter taps of the rational one ARMA filter.
    
    Inputs:
        inverse_taps (torch.tensor): array of filter taps psi_{p}^{fge}; shape:
            out_features x edge_features x denominator_order x in_features
        direct_taps (torch.tensor): array of taps varphi_{p}^{fge}; shape:
            out_features x edge_features x denominator_order x in_features
        filter_taps (torch.tensor): array of filter taps phi_{p}^{fge}; shape:
            out_features x edge_features x residue_order x in_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N} (default: None)
        tMax (int): value of t for computing the Jacobi approximation 
            (default: 5)
            
    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The inputs are:
    #   psi in F x E x P x G (inverse coefficient in order-one rational)
    #   varphi in F x E x P x G (direct coefficient in order-one rational)
    #   phi in F x E x K x G (direct filter coefficients)
    #   x in B x G x N
    #   S in E x N x N
    F = psi.shape[0] # out_features
    E = psi.shape[1] # edge_features
    P = psi.shape[2] # inverse polynomial order
    G = psi.shape[3] # in_features
    assert varphi.shape[0] == F
    assert varphi.shape[1] == E
    assert varphi.shape[2] == P
    assert varphi.shape[3] == G
    assert phi.shape[0] == F
    assert phi.shape[1] == E
    assert phi.shape[3] == G
    B = x.shape[0] # batch_size
    assert x.shape[1] == G
    N = x.shape[2] # number_nodes
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N
    
    # First, let's build Stilde and Sbar
    Stilde = torch.empty(0).to(S.device) # Will be of shape E x N x N
    DiagS = torch.empty(0).to(S.device) # Will be of shape E x N x N
    for e in range(E):
        thisS = torch.index_select(S,0,torch.tensor(e).to(S.device)).squeeze(0)
        thisDiagS = torch.diag(torch.diag(thisS))
        DiagOffS = (thisS - thisDiagS).unsqueeze(0) # E x N x N
        Stilde = torch.cat((Stilde, DiagOffS), dim = 0)
        DiagS = torch.cat((DiagS, thisDiagS.unsqueeze(0)), dim = 0)
    I = torch.eye(N).reshape([1, 1, 1, 1, N, N]).to(S.device) # (FxExPxGxNxN)
    psiI = psi.reshape([F, E, P, G, 1, 1]) * I
    DiagS = DiagS.reshape([1, E, 1, 1, N, N])
    Sbar = DiagS - psiI # F x E x P x G x N x N
    
    # Now, invert Sbar, that doesn't depend on t either, and multiply it by x
    # Obs.: We cannot just do 1/Sbar, because all the nonzero elements will
    # give inf, ruining everything. So we will force the off-diagonal elements
    # to be one, and then get rid of them
    offDiagonalOnes = (torch.ones(N,N) - torch.eye(N)).to(Sbar.device)
    SbarInv = 1/(Sbar + offDiagonalOnes) # F x E x P x G x N x N
    SbarInv = SbarInv * torch.eye(N).to(Sbar.device)
    SbarInvX = torch.matmul(SbarInv.reshape([1, F, E, P, G, N, N]),
                            x.reshape([B, 1, 1, 1, G, N, 1])).squeeze(6)
    #   B x F x E x P x G x N
    # And also multiply SbarInv with Stilde which is also used in H1 and H2
    SbarInvStilde = torch.matmul(SbarInv,
                                 Stilde.reshape([1, E, 1, 1, N, N]))
    #   B x F x E x P x G x N x N
    
    # Next, filtering through H^{3}(S) also doesn't depend on t or p, so
    H3x = LSIGF(phi, S, x)
    
    # Last, build the output from combining all filters H1, H2 and H3
        
    # Compute H1 SbarInvX
    z = SbarInvX.reshape([B, F, E, 1, P, G, N]) 
    y = x.reshape([B, 1, 1, 1, G, N, 1]) # (B x F x E x P x G x N x 1)
    x1 = SbarInvX.unsqueeze(6) # B x F x E x P x G x N x 1
    #   (B x F x E x tau x P x G x N)
    for tau in range(1,tMax+1):
        x1 = torch.matmul(SbarInvStilde.unsqueeze(0),# 1 x F x E x P x G x N x N
                          x1)
        #   B x F x E x P x G x N x 1
        z = torch.cat((z, x1.squeeze(6).unsqueeze(3)), dim = 3)
        #   B x F x E x tau x P x G x N
        y = torch.matmul(SbarInvStilde.unsqueeze(0), # 1 x F x E x P x G x N x N
                         y) 
        # B x F x E x P x G x N x 1
    thisCoeffs = torch.tensor((-1.) ** np.arange(0,tMax+1)).to(x.device)
    thisCoeffs = thisCoeffs.reshape([1, 1, 1, tMax+1, 1, 1]) * \
                    varphi.reshape([1, F, E, 1, P, G])\
                                .repeat(1, 1, 1, tMax+1, 1, 1)
    #   1 x F x E x (tMax+1) x P x G
    thisCoeffs = thisCoeffs.permute(0, 4, 1, 2, 3, 5) 
    #   1 x P x F x E x (tMax+1) x G
    z = z.permute(0, 4, 1, 6, 2, 3, 5) # B x P x F x N x E x (tMax+1) x G
    thisCoeffs = thisCoeffs.reshape([1, P, F, E*(tMax+1)*G]).unsqueeze(4)
    #   1 x P x F x E(tMax+1)G x 1
    z = z.reshape(B, P, F, N, E*(tMax+1)*G)
    #   B x P x F x N x E*(tMax+1)*G
    H1x = torch.matmul(z, thisCoeffs).squeeze(4)
    #   B x P x F x N
    # Now, to compute H2x we need y, but y went only up to value tMax, and
    # we need to go to tMax+1, so we need to multiply it once more
    y = torch.matmul(SbarInvStilde.unsqueeze(0), y).squeeze(6)
    #   B x F x E x P x G x N
    H2x = -y if np.mod(tMax,2) == 0 else y
    H2x = torch.sum(H2x, dim = 4) # sum over G, shape: B x F x E x P x N
    H2x = torch.sum(H2x, dim = 2) # sume over E, shape: B x F x P x N
    H2x = H2x.permute(0, 2, 1, 3) # B x P x F x N
    # Finally, we add up H1x and H2x and sum over all p, and add to H3 to
    # update u
    u = torch.sum(H1x + H2x, dim = 1) + H3x
       
    if b is not None:
        u = u+b
    return u

def learnAttentionGSO(x, a, W, S, negative_slope=0.2):
    """
    learnAttentionGSO(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.
    
    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2] # output_features
    assert a.shape[2] == int(2*F)
    G = W.shape[3] # input_features
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    S = S.type(torch.float) + torch.eye(N).reshape([1,N,N]).repeat(E,1,1).to(S.device)
    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x) # B x P x E x F x N
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device)) # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2*F).to(x.device)) # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx) # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx) # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3) # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    eij = nn.functional.leaky_relu(aWx, negative_slope = negative_slope)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim = 0)
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype)
    #   Make it -infinity where there are zeros
    infinityMask = (1-maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij*maskEdges - infinityMask, dim = 4)
    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    return aij * maskEdges # B x P x E x N x N


def learnAttentionGSOBatch(x, a, W, W_b, S, negative_slope=0.2):
    """ v1
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x)  # B x P x E x F x N
    Wx = Wx + W_b.reshape([1, P, E, F, 1]).repeat(B, 1, 1, 1, N)
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device))  # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2 * F).to(x.device))  # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3)  # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu

    eij = nn.functional.leaky_relu(aWx, negative_slope=negative_slope)
    # eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    # S_all_one = torch.ones([B, E, N, N]).to(x.device)
    # maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # print("S",S)
    # print("aij",aij)

    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)


def learnAttentionGSOBatch_LSTM(x, a, W, W_b, S, LSTM_layer=None, negative_slope=0.2):
    """ v1
    learnAttentionGSOBatch_LSTM(x, a, W, W_b, S, LSTM_layer=None, negative_slope=0.2)
    Computes the GSO following the attention mechanism with LSTM

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T LSTM( [cat(W^{ep}x_{i}, W^{ep} x_{j}) )[-1]]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    assert LSTM_layer is not None
    T = x.shape[0]  # SeqLength
    B = x.shape[1]  # batch_size
    G = x.shape[2]  # input_features
    N = x.shape[3]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([T, B, 1, 1, G, N])
    W = W.reshape([1, 1, P, E, F, G])
    Wx = torch.matmul(W, x)  # T x B x P x E x F x N
    Wx = Wx + W_b.reshape([1, P, E, F, 1]).repeat(T, B, 1, 1, 1, N) # T x B x P x E x F x N
    
    # Form i -> j concatenated matrix
    # T x B x P x E x N x FN
    repeated_Wx_1 = Wx.repeat_interleave(N, dim=-1)
    repeated_Wx_2 = Wx.repeat(1,1,1,1,1,N)
    concat_Wx = torch.cat([repeated_Wx_1, repeated_Wx_2], dim=-2) # T x B x P x E x 2F x N*N

    concat_Wx = concat_Wx.reshape(T, B*P*E, F*2, N*N).permute(0, 1, 3, 2).reshape(T, B*P*E*N*N, F*2) # T x BPEN*N x 2F


    #  ====  Using LSTM  =====
    # print(concat_Wx.shape)
    LSTM_output, LSTM_hn = LSTM_layer(concat_Wx) # T x BPENN x 1
    # print(LSTM_output.shape)

    # TODO: Test on two solutions

    # Sol 1: output x a matrix
    # Multiply by mixer a to shape into 1 value
    # LSTM_output = LSTM_output.reshape(T, B, P, E, N*N, F*2)[-1].permute(0,1,2,4,3) # B x P x E x 2F x N*N
    # print(LSTM_output.shape)
    # a_reshaped = a.reshape([1, P, E, 1, 2*F])
    # LSTM_output = concat_Wx
    # output = torch.matmul(a_reshaped, LSTM_output) # B x P x E x 1 x N*N


    # Sol 2: output attention directly
    output = LSTM_output[-1].reshape((B, P, E, N*N, 1)).permute(0,1,2,4,3) # B x P x E x 1 x N*N

    # Unpack i->j correspondence
    output = output.reshape(B, P, E, N*N)
    # B x P x E x N*N
    output = output.reshape(B, P, E, N, N).permute(0, 1, 2, 4, 3)
    # B x P x E x N x N

    # Apply the LeakyRelu
    eij = nn.functional.leaky_relu(output, negative_slope=negative_slope)
    # eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    # S_all_one = torch.ones([B, E, N, N]).to(x.device)
    # maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # print("S",S)
    # print("aij",aij)

    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)


def learnAttentionGSOBatch_origin(x, a, W, S, negative_slope=0.2):
    """
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x)  # B x P x E x F x N
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device))  # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2 * F).to(x.device))  # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3)  # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    eij = nn.functional.leaky_relu(aWx, negative_slope=negative_slope)
    # eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    # S_all_one = torch.ones([B,E, N,N]).to(x.device)
    # maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)

def learnAttentionGSOBatch_gso_allone_v2(x, a, W, S, negative_slope=0.2):
    """ v2
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x)  # B x P x E x F x N
    # Now, do a_1^T Wx, and a_2^T Wx to get a tensor of shape B x P x E x 1 x N
    # because we're applying the inner product on the F dimension.
    a1 = torch.index_select(a, 2, torch.arange(F).to(x.device))  # K x E x F
    a2 = torch.index_select(a, 2, torch.arange(F, 2 * F).to(x.device))  # K x E x F
    a1Wx = torch.matmul(a1.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    a2Wx = torch.matmul(a2.reshape([1, P, E, 1, F]), Wx)  # B x P x E x 1 x N
    # And then, use this to sum them accordingly and create a B x P x E x N x N
    # matrix.
    aWx = a1Wx + a2Wx.permute(0, 1, 2, 4, 3)  # B x P x E x N x N
    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    # eij = nn.functional.leaky_relu(aWx, negative_slope=negative_slope)
    eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    S_all_one = torch.ones([B,E, N,N]).to(x.device)
    maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    # maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    # return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)

def learnAttentionGSOBatch_KeyQuery(x, a, W, W_b, S, negative_slope=0.2):
    """
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism
    https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc

    Multiplicative Attention:
        eij = q * W * k

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x input_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    # F = W.shape[2]  # output_features
    # assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    # x_key = x.reshape([B, 1, 1, N, G])

    # x = batch_size x input_features x number_nodes -> x = batch_size x number_nodes x input_features
    x_key = x.permute([0,2,1]).reshape([B, 1, 1, N, G])
    # x_key = x.reshape([B, 1, 1, N, G])
    x_query = x_key.permute([0,1,2,4,3]) # x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, G, G])

    # # B x P x E x N x G * B x 1 x 1 x Gx G
    # # = B x P x E x N x G
    # Wx = torch.matmul(x_key, W)

    # B x P x E x G x G * B x 1 x 1 x Gx M
    # = B x P x E x G x N
    Wx = torch.matmul(W, x_query)

    # B x P x E x N x G * B x 1 x 1 x Gx N
    # = B x P x E x N x N
    # xWx = torch.matmul(Wx, x_query) # B x P x E x N x N
    xWx = torch.matmul(x_key, Wx)
    # Wx = Wx + W_b.reshape([1, P, E, G, 1]).repeat(B, 1, 1, 1, N)

    eij = xWx
    # eij = nn.functional.leaky_relu(xWx, negative_slope=negative_slope)
    #   B x P x E x N x N

    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    # print("eij", eij.shape)
    # print("maskEdges", maskEdges.shape)

    aij_tmp = nn.functional.softmax(eij * maskEdges - infinityMask, dim=4)

    return aij_tmp * maskEdges

    ##########################################
    ########        add threshold to filter out very small weight
    ##########################################

    # threshold = np.divide(1, N)
    # mask_threshold = (aij_tmp > threshold).type(x.dtype)
    # infinityMask_threshold = (1 - mask_threshold) * infiniteNumber
    #
    #
    # aij = aij_tmp * mask_threshold

    ## Use second softmax to avoid discontinuity
    # aij = nn.functional.softmax(aij_tmp * mask_threshold - infinityMask_threshold, dim=4)

    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * maskEdges  # B x P x E x N x N


def kl_div_dim(x1,x2,dim=1,eps_min=zeroTolerance, eps_max=infiniteNumber):
    r"""Returns Kullback–Leibler divergence between x1 and x2, computed along dim.
    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero. Default: 1e-9
    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
        https://stackoverflow.com/questions/49886369/kl-divergence-for-two-probability-distributions-in-pytorch
    """
    x1_c = x1.clamp(min=eps_min, max=eps_max)
    x2_c = x2.clamp(min=eps_min, max=eps_max)
    result = (x1_c * (x1_c / x2_c).log()).sum(dim=dim)
    return result

def learnAttentionGSOBatch_DualHead(x, a, W, W_b, S, negative_slope=0.2):
    """ v1
    learnAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = int(a.shape[0]/2)  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == 2*P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    # S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, 2*P, E, F, G])

    W_cos = torch.index_select(W, 1, torch.arange(P).type(torch.int64).to(x.device))  # P x K x E x F
    W_kl = torch.index_select(W, 1, torch.arange(P, 2*P).type(torch.int64).to(x.device))  # P x K x E x F

    ## Similarity Head
    cos_weight = nn.CosineSimilarity(dim=3, eps=zeroTolerance)
    Wx_cos = torch.matmul(W_cos, x)  # B x P x E x F x N
    Wx_cos1 = Wx_cos.reshape([B, P, E, F, N, 1])
    Wx_cos2 = Wx_cos.reshape([B, P, E, F, 1, N]) # B x P x E x F x N

    aWx_cos = cos_weight(Wx_cos1, Wx_cos2) # B x P x E x N x N

    ## DisSimilarity Head
    Wx_kl = torch.matmul(W_kl, x)  # B x P x E x F x N
    Wx_kl1 = Wx_kl.reshape([B, P, E, F, N, 1])
    Wx_kl2 = Wx_kl.reshape([B, P, E, F, 1, N]) # B x P x E x F x N

    # aWx_kl = F.kl_div(Wx1.log(), Wx2, size_average=False, reduce=False, reduction='none')
    aWx_kl = kl_div_dim(Wx_kl1, Wx_kl2, dim=3) # B x P x E x N x N

    #TODO concate them togeher and compute weight

    aWx = torch.cat([aWx_cos, aWx_kl], dim=1)  # B x 2P x E x N x N

    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu
    # eij = nn.functional.leaky_relu(aWx, negative_slope=negative_slope)
    # eij = torch.ones([B, P, E, N, N]).to(x.device)
    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    # S_all_one = torch.ones([B,E, N,N]).to(x.device)
    # maskEdges = torch.sum(torch.abs(S_all_one.data), dim=1).reshape([B, 1, 1, N, N])

    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    # B x 2P x E x N x N
    aij = nn.functional.softmax(aWx * maskEdges - infinityMask, dim=4)
    #   B x 2P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    # return aij * S.reshape([B, 1, 1, N, N]).type(torch.float)
    return aij * maskEdges  # B x P x E x N x N
    # return S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)


def learnSimilarityAttentionGSOBatch(x, a, W, W_b, S, negative_slope=0.2):
    """
    learnSimilarityAttentionGSOBatch(x, a, W, S) Computes the GSO following the attention
        mechanism

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Each elements of the new GSO is alpha_{ij}^{ep} computed as
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        aij: output GSO; shape:
         batch_size x number_heads x edge_features x number_nodes x number_nodes
    """
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # output_features
    assert a.shape[2] == int(2 * F)
    G = W.shape[3]  # input_features
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N
    # assert S.shape[0] == E
    # assert S.shape[1] == S.shape[2] == N

    # Add ones of the GSO at all edge feature levels so that the node always
    # has access to itself. The fact that it's one is not so relevant, because
    # the attention coefficient that is learned would compensate for this
    S = S.type(torch.float) + torch.eye(N, dtype=torch.float).reshape([1, N, N]).repeat(B, E, 1, 1).to(S.device)

    # WARNING:
    # (If the GSOs already have self-connections, then these will be added a 1,
    # which might be a problem if the self-connection is a -1. I will have to
    # think of this more carefully)

    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])

    cos_weight = nn.CosineSimilarity(dim=3, eps=zeroTolerance)
    # aWx_cos = cos_weight(x1, x1)  #B x P x E  x N X N

    # W_cos = W.reshape([1, P, E, F, G])
    # Wx_cos = torch.matmul(W_cos, x)  # B x P x E x F x N
    # Wx_cos1 = Wx_cos.reshape([B, P, E, F, N, 1])
    # Wx_cos2 = Wx_cos.reshape([B, P, E, F, 1, N]) # B x P x E x F x N
    # aWx_cos = cos_weight(Wx_cos1, Wx_cos2) # B x P x E x N x N

    # version 2

    # 1 x P x E x G x G
    W = W.reshape([1, P, E, G, G])
    # 1 x P x E x G x G
    # B x 1 x 1 x G x N
    Wx = torch.matmul(W, x)  # B x P x E x G x N

    x_new = x.reshape([B, P, E, G, N, 1])

    Wx_new = Wx.reshape([B, P, E, G, 1, N])

    aWx_cos = cos_weight(x_new, Wx_new) # B x P x E x N x N

    # aWx = torch.matmul(a.reshape([1, P, E, 1, F]), Wx)

    #   Obs.: In this case, we have one column vector and one row vector; then,
    # what the sum does, is to repeat the column and the row, respectively,
    # until both matrices are of the same size, and then adds up, which is
    # precisely what we want to do
    # Apply the LeakyRelu

    #   B x P x E x N x N
    # Each element of this N x N matrix is, precisely, e_ij (eq. 1) in the GAT
    # paper.
    # And apply the softmax. For the softmax, we do not want to consider
    # the places where there are no neighbors, so we need to set them to -infty
    # so that they will be assigned a zero.
    #   First, get places where we have edges
    maskEdges = torch.sum(torch.abs(S.data), dim=1).reshape([B, 1, 1, N, N]) # B x 1 x 1 x N x N
    #   Make it a binary matrix
    maskEdges = (maskEdges > zeroTolerance).type(x.dtype) # B x 1 x 1 x N x N
    #   Make it -infinity where there are zeros
    infinityMask = (1 - maskEdges) * infiniteNumber
    #   Compute the softmax plus the -infinity (we first force the places where
    # there is no edge to be zero, and then we add -infinity to them)
    aij = nn.functional.softmax(aWx_cos * maskEdges - infinityMask, dim=4)
    #   B x P x E x N x N
    # This will give me a matrix of all the alpha_ij coefficients.
    # Re-inforce the zeros just to be sure
    return aij * maskEdges  # B x P x E x N x N

def graphAttention(x, a, W, S, negative_slope=0.2):
    """
    graphAttention(x, a, W, S) Computes attention following GAT layer taking
        into account multiple edge features.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, P the number of attention heads, Ji the
    number of nodes in N_{i}, the neighborhood of node i, and N the number of
    nodes. Let x_{i} in R^{G} be the feature associated to node i,
    W^{ep} in R^{F x G} the weight marix associated to edge feature e and
    attention head p, and a^{ep} in R^{2F} the mixing vector. Let
    alpha_{ij}^{ep} in R the attention coefficient between nodes i and j, for
    edge feature e and attention head p, and let s_{ij}^{e} be the value of
    feature e of the edge connecting nodes i and j.

    Let y_{i}^{p} in R^{F} be the output of the graph attention at node i for
    attention head p. It is computed as
        y_{i}^{p} = \sum_{e=1}^{E}
                        \sum_{j in N_{i}}
                            s_{ij}^{e} alpha_{ij}^{ep} W^{ep} x_{j}
    with
        alpha_{ij}^{ep} = softmax_{j} ( LeakyReLU_{beta} (
                (a^{ep})^T [cat(W^{ep}x_{i}, W^{ep} x_{j})]
        ))
    for all j in N_{i}, and where beta is the negative slope of the leaky ReLU.

    Inputs:
        x (torch.tensor): input;
            shape: batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * output_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x output_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2] # output_features
    assert a.shape[2] == int(2*F)
    G = W.shape[3] # input_features
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # First, we need to learn the attention GSO
    aij = learnAttentionGSO(x, a, W, S, negative_slope = negative_slope)
    # B x P x E x N x N

    # Then, we need to compute the high-level features
    # W is of size P x E x F x G
    # a is of size P x E x 2F
    # Compute Wx for all nodes
    x = x.reshape([B, 1, 1, G, N])
    W = W.reshape([1, P, E, F, G])
    Wx = torch.matmul(W, x) # B x P x E x F x N
    
    # Finally, we just need to apply this matrix to the Wx which we have already
    # computed, and done.
    y = torch.matmul(Wx, S.reshape([1, 1, E, N, N]) * aij) # B x P x E x F x N
    # And sum over all edges
    return torch.sum(y, dim = 2) # B x P x F x N

def graphAttentionLSIGF(h, x, a, W, S, b=None, negative_slope=0.2):
    """
    graphAttentionLSIGF(h, x, a, W, S) Computes a graph convolution 
        (LSIGF) over a graph shift operator learned through the attention
        mechanism

    Inputs:
        h (torch.tensor): array of filter taps; shape:
            edge_features x filter_taps
        x (torch.tensor): input; shape:
            batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * out_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x out_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    E = h.shape[0] # edge_features
    K = h.shape[1] # filter_taps
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    E = a.shape[1] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2] # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2*F)
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # First, we need to learn the attention GSO
    aij = learnAttentionGSO(x, a, W, S, negative_slope = negative_slope)
    # B x P x E x N x N
    
    # And now we need to compute an LSIGF with this learned GSO, but the filter
    # taps of the LSIGF are a combination of h (along K), and W (along F and G)
    # So, we have
    #   h in E x K
    #   W in P x E x G x F
    # The filter taps, will thus have shape
    #   h in P x F x E x K x G
    h = h.reshape([1, 1, E, K, 1]) # (P x F x E x K x G)
    W = W.permute(0, 3, 1, 2) # P x F x E x G
    W = W.reshape([P, F, E, 1, G]) # (P x F x E x K x G)
    h = h * W # P x F x E x K x G (We hope, if not, we need to repeat on the 
        # corresponding dimensions)
    
    x = x.reshape([B, 1, 1, G, N]) # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the 
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1,K):
        x = torch.matmul(x, aij) # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N]) # add the k dimension
        z = torch.cat((z, xAij), dim = 3) # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E*K*G]) # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2) # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E*K*G])
    # And multiply
    y = torch.matmul(z, h) # B x P x N x F
    y = y.permute(0, 1, 3, 2) # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y+b
    return y

def graphAttentionLSIGFBatch_KeyQuery(h, x, a, W, W_b, S, b=None, negative_slope=0.2):

    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps
    F = h.shape[4]  # out_features

    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    assert W.shape[3] == G
    # assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch_KeyQuery(x, a, W, W_b, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # h: P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij

def graphAttentionLSIGFBatch_modified(h, x, a, W, W_b, S, b=None, negative_slope=0.2):

    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch(x, a, W, W_b, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # h: P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij

def graphAttentionLSIGFBatch_modified_LSTM(h, x, a, W, W_b, S, b=None, LSTM_layer=None, negative_slope=0.2):
    assert LSTM_layer is not None

    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps

    T = x.shape[0]  # SeqLength
    B = x.shape[1]  # batch_size
    G = x.shape[2]  # input_features
    N = x.shape[3]  # number_nodes

    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch_LSTM(x, a, W, W_b, S, LSTM_layer=LSTM_layer, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # h: P x F x E x K x G
    x = x[-1].reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij



def graphAttentionLSIGFBatch_DualHead(h, x, a, W, W_b, S, b=None, negative_slope=0.2):

    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = int(a.shape[0]/2)  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == 2*P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch_DualHead(x, a, W, W_b, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x 2 P x E x N x N

    # h: 2P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, 2*P, E, 1, 1, 1)
    # add the k=0 dimension (B x 2P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x 2P x E x G x N
        xAij = x.reshape([B, 2*P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x 2 P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, 2*P, F, E * K * G])  # (B x 2P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x 2P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, 2*P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x 2P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x 2P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij

def graphAttentionLSIGFBatch_Origin(h, x, a, W, S, b=None, negative_slope=0.2):

    E = h.shape[0]  # edge_features
    K = h.shape[1]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    aij = learnAttentionGSOBatch_origin(x, a, W, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # And now we need to compute an LSIGF with this learned GSO, but the filter
    # taps of the LSIGF are a combination of h (along K), and W (along F and G)
    # So, we have
    #   h in E x K
    #   W in P x E x G x F
    # The filter taps, will thus have shape
    #   h in P x F x E x K x G
    h = h.reshape([1, 1, E, K, 1])  # (P x F x E x K x G)
    W = W.permute(0, 3, 1, 2)  # P x F x E x G
    W = W.reshape([P, F, E, 1, G])  # (P x F x E x K x G)
    h = h * W  # P x F x E x K x G (We hope, if not, we need to repeat on the
    # corresponding dimensions)

    # h: P x F x E x K x G
    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij




def graphAttentionLSIGFBatch(h, x, a, W, S, b=None, negative_slope=0.2):
    """
    graphAttentionLSIGF(h, x, a, W, S) Computes a graph convolution
        (LSIGF) over a graph shift operator learned through the attention
        mechanism

    Inputs:
        h (torch.tensor): array of filter taps; shape:
            edge_features x filter_taps
        x (torch.tensor): input; shape:
            batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * out_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x out_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    E = h.shape[0]  # edge_features
    K = h.shape[1]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    # First, we need to learn the attention GSO
    aij = learnAttentionGSOBatch(x, a, W, S, negative_slope=negative_slope)
    # aij = S.reshape([B, 1, 1, N, N]).repeat(1, P, E, 1, 1).type(torch.float)
    # B x P x E x N x N

    # And now we need to compute an LSIGF with this learned GSO, but the filter
    # taps of the LSIGF are a combination of h (along K), and W (along F and G)
    # So, we have
    #   h in E x K
    #   W in P x E x G x F
    # The filter taps, will thus have shape
    #   h in P x F x E x K x G
    h = h.reshape([1, 1, E, K, 1])  # (P x F x E x K x G)
    W = W.permute(0, 3, 1, 2)  # P x F x E x G
    W = W.reshape([P, F, E, 1, G])  # (P x F x E x K x G)
    h = h * W  # P x F x E x K x G (We hope, if not, we need to repeat on the
    # corresponding dimensions)

    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij


def graphSimilarityAttentionLSIGFBatch(h, x, a, W, W_b, S, b=None, negative_slope=0.2):
    """
    graphAttentionLSIGF(h, x, a, W, S) Computes a graph convolution
        (LSIGF) over a graph shift operator learned through the attention
        mechanism

    Inputs:
        h (torch.tensor): array of filter taps; shape:
            edge_features x filter_taps
        x (torch.tensor): input; shape:
            batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x edge_features x 2 * out_features
        W (torch.tensor): linear parameter; shape:
            number_heads x edge_features x out_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    E = h.shape[2]  # edge_features
    K = h.shape[3]  # filter_taps
    B = x.shape[0]  # batch_size
    G = x.shape[1]  # input_features
    N = x.shape[2]  # number_nodes
    P = a.shape[0]  # number_heads
    E = a.shape[1]  # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == E
    F = W.shape[2]  # out_features
    assert W.shape[3] == G
    assert a.shape[2] == int(2 * F)
    assert S.shape[1] == E
    assert S.shape[2] == S.shape[3] == N

    # First, we need to learn the attention GSO
    aij = learnSimilarityAttentionGSOBatch(x, a, W, W_b, S, negative_slope=negative_slope)
    # B x P x E x N x N

    # And now we need to compute an LSIGF with this learned GSO, but the filter
    # taps of the LSIGF are a combination of h (along K), and W (along F and G)
    # So, we have
    #   h in E x K
    #   W in P x E x G x F
    # The filter taps, will thus have shape
    #   h in P x F x E x K x G
    # h = h.reshape([1, 1, E, K, 1])  # (P x F x E x K x G)
    # W = W.permute(0, 3, 1, 2)  # P x F x E x G
    # W = W.reshape([P, F, E, 1, G])  # (P x F x E x K x G)
    # h = h * W  # P x F x E x K x G (We hope, if not, we need to repeat on the
    # corresponding dimensions)

    x = x.reshape([B, 1, 1, G, N])  # (B x P x E x G x N)
    # The easiest would be to use the LSIGF function, but that takes as input
    # a B x F x N input, and while we could join together B and P into a single
    # dimension, we would still be unable to handle the E features this way.
    # So we basically need to copy the code from LSIGF but accounting the
    # matrix multiplications with multiple edge features as Wx has
    z = x.reshape([B, 1, 1, 1, G, N]).repeat(1, P, E, 1, 1, 1)
    # add the k=0 dimension (B x P x E x K x G x N)
    # And now do the repeated multiplication with S
    for k in range(1, K):
        x = torch.matmul(x, aij)  # B x P x E x G x N
        xAij = x.reshape([B, P, E, 1, G, N])  # add the k dimension
        z = torch.cat((z, xAij), dim=3)  # B x P x E x k x G x N
    # This output z is of shape B x P x E x K x M x N and represents the product
    # x * aij_{e}^{k} (i.e. the multiplication between x and the kth power of
    # the learned GSO).
    # Now, we need to multiply this by the filter coefficients
    # Convert h, from F x E x K x M to EKM x F to multiply from the right
    h = h.reshape([1, P, F, E * K * G])  # (B x P x F x (EKG))
    h = h.permute(0, 1, 3, 2)  # (B x P x EKG x F)
    # And z from B x P x E x K x G x N to B x P x N x EKG to left multiply
    z = z.permute(0, 1, 5, 2, 3, 4).reshape([B, P, N, E * K * G])
    # And multiply
    y = torch.matmul(z, h)  # B x P x N x F
    y = y.permute(0, 1, 3, 2)  # The output needs to be B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y, aij

def graphAttentionEVGF(x, a, W, S, b=None, negative_slope=0.2):
    """
    graphAttentionEVGF(h, x, a, W, S) Computes an edge varying graph filter
        (EVGF) where each EVGF is learned by an attention mechanism

    Inputs:
        x (torch.tensor): input; shape:
            batch_size x input_features x number_nodes
        a (torch.tensor): mixing parameter; shape:
            number_heads x filter_taps x edge_features x 2 * out_features
        W (torch.tensor): linear parameter; shape:
            number_heads x filter_taps x edge_features x out_features x input_features
        S (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
        negative_slope (float): negative slope of the leaky relu (default: 0.2)

    Outputs:
        y: output; shape:
            batch_size x number_heads x output_features x number_nodes
    """
    B = x.shape[0] # batch_size
    G = x.shape[1] # input_features
    N = x.shape[2] # number_nodes
    P = a.shape[0] # number_heads
    K = a.shape[1] # filter_taps
    E = a.shape[2] # edge_features
    assert W.shape[0] == P
    assert W.shape[1] == K
    assert W.shape[2] == E
    F = W.shape[3] # output_features
    assert W.shape[4] == G
    assert a.shape[3] == int(2*F)
    assert S.shape[0] == E
    assert S.shape[1] == S.shape[2] == N

    # First, we need to compute the high-level features
    # W is of size P x K x E x F x G
    # a is of size P x K x E x 2F
    # To compute Wx, we need the first element (K = 0)
    W0 = torch.index_select(W, 1, torch.tensor(0).to(S.device)).squeeze(1)
    #   P x E x F x G
    W0 = W0.reshape([1, P, E, F, G])
    W0x = torch.matmul(W0, x.reshape([B, 1, 1, G, N])) # B x P x E x F x N
    
    # Now we proceed to learn the rest of the EVGF.
    # That first filter coefficient (for the one-hop neighborhood) is learned
    # from the first element along the K dimension (dim = 1)
    thisa = torch.index_select(a, 1, torch.tensor(0).to(S.device)).squeeze(1)
    thisW = torch.index_select(W, 1, torch.tensor(0).to(S.device)).squeeze(1)
    aij = learnAttentionGSO(x, thisa, thisW, S, negative_slope = negative_slope)
    # B x P x E x N x N (repesents k=0,1)
    W0x = torch.matmul(W0x, S.reshape([1, 1, E, N, N]) * aij) # B x P x E x F x N
    y = W0x # This is the first multiplication between Wx and Aij corresponding
        # to the first-hop neighborhood
    
    # Now, we move on to the rest of the coefficients
    for k in range(1, K):
        thisa = torch.index_select(a,1, torch.tensor(k).to(S.device)).squeeze(1)
        thisW = torch.index_select(W,1, torch.tensor(k).to(S.device)).squeeze(1)
        aij = learnAttentionGSO(x, thisa, thisW, S,
                                negative_slope = negative_slope)
        W0x = torch.matmul(W0x, S.reshape([1, 1, E, N, N]) * aij)
            # This multiplies the previous W0x Aij^{1:k-1} with A_ij^{(k)}
        y = y + W0x # Adds that multiplication to the running sum for all other
            # ks, shape: B x P x E x F x N
    
    # Sum over all edge features
    y = torch.sum(y, dim = 2) # B x P x F x N
    # Finally, add the bias
    if b is not None:
        y = y+b
    return y

#############################################################################
#                                                                           #
#                  FUNCTIONALS (Batch, and time-varying)                    #
#                                                                           #
#############################################################################

def LSIGF_DB(h, S, x, b=None):
    """
    LSIGF_DB(filter_taps, GSO, input, bias=None) Computes the output of a 
        linear shift-invariant graph filter (graph convolution) on delayed 
        input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e}(t) in R^{N x N} the GSO for edge feature e at time t, 
    x(t) in R^{G x N} the input data at time t where x_{g}(t) in R^{N} is the
    graph signal representing feature g, and b in R^{F x N} the bias vector,
    with b_{f} in R^{N} representing the bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}(t)S_{e}(t-1)...S_{e}(t-(k-1))
                                        x_{g}(t-k)
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x time_samples x edge_features 
                                                  x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x time_samples x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x time_samples x output_features x number_nodes
    """
    # This is the LSIGF with (B)atch and (D)elay capabilities (i.e. there is
    # a different GSO for each element in the batch, and it handles time
    # sequences, both in the GSO as in the input signal. The GSO should be
    # transparent).
    
    # So, the input 
    #   h: F x E x K x G
    #   S: B x T x E x N x N
    #   x: B x T x G x N
    #   b: F x N
    # And the output has to be
    #   y: B x T x F x N
    
    # Check dimensions
    assert len(h.shape) == 4
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert len(S.shape) == 5
    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N
    assert len(x.shape) == 4
    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == G
    assert x.shape[3] == N
    
    # We would like a z of shape B x T x K x E x G x N that represents, for
    # each t, x_t, S_t x_{t-1}, S_t S_{t-1} x_{t-2}, ...,
    #         S_{t} ... S_{t-(k-1)} x_{t-k}, ..., S_{t} S_{t-1} ... x_{t-(K-1)}
    # But we don't want to do "for each t". We just want to do "for each k".
    
    # Let's start by reshaping x so it can be multiplied by S
    x = x.reshape([B, T, 1, G, N]).repeat(1, 1, E, 1, 1)
    
    # Now, for the first value of k, we just have the same signal
    z = x.reshape([B, T, 1, E, G, N])
    # For k = 0, k is counted in dim = 2
    
    # Now we need to start multiplying with S, but displacing the entire thing
    # once across time
    for k in range(1,K):
        # Across dim = 1 we need to "displace the dimension down", i.e. where
        # it used to be t = 1 we now need it to be t=0 and so on. For t=0
        # we add a "row" of zeros.
        x, _ = torch.split(x, [T-1, 1], dim = 1)
        #   The second part is the most recent time instant which we do not need
        #   anymore (it's used only once for the first value of K)
        # Now, we need to add a "row" of zeros at the beginning (for t = 0)
        zeroRow = torch.zeros(B, 1, E, G, N, dtype=x.dtype,device=x.device)
        x = torch.cat((zeroRow, x), dim = 1)
        # And now we multiply with S
        x = torch.matmul(x, S)
        # Add the dimension along K
        xS = x.reshape(B, T, 1, E, G, N)
        # And concatenate it with z
        z = torch.cat((z, xS), dim = 2)
        
    # Now, we finally made it to a vector z of shape B x T x K x E x G x N
    # To finally multiply with the filter taps, we need to swap the sizes
    # and reshape
    z = z.permute(0, 1, 5, 3, 2, 4) # B x T x N x E x K x G
    z = z.reshape(B, T, N, E*K*G)
    # And the same with the filter taps
    h = h.reshape(F, E*K*G)
    h = h.permute(1, 0) # E*K*G x F
    
    # Multiply
    y = torch.matmul(z, h) # B x T x N x F
    # And permute
    y = y.permute(0, 1, 3, 2) # B x T x F x N
    # Finally, add the bias
    if b is not None:
        y = y + b
    
    return y

def GRNN_DB(a, b, S, x, z0, sigma,
                   xBias=None, zBias = None):
    """
    GRNN_DB(signal_to_hidden_taps, hidden_to_hidden_taps, GSO, input,
            initial_hidden, nonlinearity, signal_bias, hidden_bias)
    Computes the sequence of hidden states for the input sequence x following 
    the equation z_{t} = sigma(A(S) x_{t} + B(S) z_{t-1}) with initial state z0
    and where sigma is the nonlinearity, and A(S) and B(S) are the 
    Input-to-Hidden filters and the Hidden-to-Hidden filters with the 
    corresponding taps.
    
    Inputs:
        signal_to_hidden_taps (torch.tensor): shape
            hidden_features x edge_features x filter_taps x signal_features
        hidden_to_hidden_taps (torch.tensor): shape
            hidden_features x edge_features x filter_taps x hidden_features
        GSO (torch.tensor): shape
            batch_size x time x edge_features x number_nodes x number_nodes
        input (torch.tensor): shape
            batch_size x time x signal_features x number_nodes
        initial_hidden: shape
            batch_size x hidden_features x number_nodes
        signal_bias (torch.tensor): shape
            1 x 1 x hidden_features x 1
        hidden_bias (torch.tensor): shape
            1 x 1 x hidden_features x 1
    
    Outputs:
        hidden_state: shape
            batch_size x time x hidden_features x number_nodes
            
    """
    # We will compute the hidden state for a delayed and batch data.
    
    # So, the input
    #   a: H x E x K x F (Input to Hidden filters)
    #   b: H x E x K x H (Hidden to Hidden filters)
    #   S: B x T x E x N x N (GSO)
    #   x: B x T x F x N (Input signal)
    #   z0: B x H x N (Initial state)
    #   xBias: 1 x 1 x H x 1 (bias on the Input to Hidden features)
    #   zBias: 1 x 1 x H x 1 (bias on the Hidden to Hidden features)
    # And the output has to be
    #   z: B x T x H x N (Hidden state signal)
    
    # Check dimensions
    H = a.shape[0] # Number of hidden state features
    E = a.shape[1] # Number of edge features
    K = a.shape[2] # Number of filter taps
    F = a.shape[3] # Number of input features
    assert b.shape[0] == H
    assert b.shape[1] == E
    assert b.shape[2] == K
    assert b.shape[3] == H
    B = S.shape[0]
    T = S.shape[1]
    assert S.shape[2] == E
    N = S.shape[3]
    assert S.shape[4] == N
    assert x.shape[0] == B
    assert x.shape[1] == T
    assert x.shape[2] == F
    assert x.shape[3] == N
    assert z0.shape[0] == B
    assert z0.shape[1] == H
    assert z0.shape[2] == N
    
    # The application of A(S) x(t) doesn't change (it does not depend on z(t))
    Ax = LSIGF_DB(a, S, x, b = xBias) # B x T x H x N
    # This is the filtered signal for all time instants.
    # This also doesn't split S, it only splits x.
    
    # The b parameters we will always need them in this shape
    b = b.unsqueeze(0).reshape(1, H, E*K*H) # 1 x H x EKH
    # so that we can multiply them with the product Sz that should be of shape
    # B x EKH x N
    
    # We will also need a selection matrix that selects the first K-1 elements
    # out of the original K (to avoid torch.split and torch.index_select with
    # more than one index)
    CK = torch.eye(K-1, device = S.device) # (K-1) x (K-1)
    zeroRow = torch.zeros((1, K-1), device = CK.device)
    CK = torch.cat((CK, zeroRow), dim = 0) # K x (K-1)
    # This matrix discards the last column when multiplying on the left
    CK = CK.reshape(1, 1, 1, K, K-1) # 1(B) x 1(E) x 1(H) x K x K-1
    
    #\\\ Now compute the first time instant
    
    # We just need to multiplicate z0 = z(-1) by b(0) to get z(0)
    #   Create the zeros that will multiply the values of b(1), b(2), ... b(K-1)
    #   since we only need b(0)
    zerosK = torch.zeros((B, K-1, H, N), device = z0.device)
    #   Concatenate them after z
    zK = torch.cat((z0.unsqueeze(1), zerosK), dim = 1) # B x K x H x N
    #   Now we have a signal that has only the z(-1) and the rest are zeros, so
    #   now we can go ahead and multiply it by b. For this to happen, we need
    #   to reshape it as B x EKH x N, but since we are always reshaping the last
    #   dimensions we will bring EKH to the end, reshape, and then put them back
    #   in the middle.
    zK = zK.reshape(B, 1, K, H, N).repeat(1, E, 1, 1, 1) # B x E x K x H x N
    zK = zK.permute(0, 4, 1, 2, 3).reshape(B, N, E*K*H).permute(0, 2, 1)
    #   B x EKH x N
    #   Finally, we can go ahead an multiply with b
    zt = torch.matmul(b, zK) # B x H x N
    # Now that we have b(0) z(0) we can add the bias, if necessary
    if zBias is not None:
        zt = zt + zBias
    # And we need to add it to a(0)x(0) which is the first element of Ax in the
    # T dimension
    # Let's do a torch.index_select; not so sure a selection matrix isn't better
    a0x0 = torch.index_select(Ax, 1, torch.tensor(0, device = Ax.device)).reshape(B, H, N)
    #   B x H x N
    # Recall that a0x0 already has the bias, so now we just need to add up and
    # apply the nonlinearity
    zt = sigma(a0x0 + zt) # B x H x N
    z = zt.unsqueeze(1) # B x 1 x H x N
    zt = zt.unsqueeze(1) # B x 1 x H x N
    # This is where we will keep track of the product Sz
    Sz = z0.reshape(B, 1, 1, H, N).repeat(1, 1, E, 1, 1) # B x 1 x E x H x N
    
    # Starting now, we need to multiply this by S every time
    for t in range(1,T):
        if t < K:
            # Get the current time instant
            St = torch.index_select(S, 1, torch.tensor(t, device = S.device))
            #   B x 1 x E x N x N
            # We need to multiply this time instant by all the elements in Sz
            # now, and there are t of those
            St = St.repeat(1, t, 1, 1, 1) # B x t x E x N x N
            # Multiply by the newly acquired St to do one more delay
            Sz = torch.matmul(Sz, St) # B x t x E x H x N
            # Observe that these delays are backward: the last element in the
            # T dimension (dim = 1) is the latest element, this makes sense 
            # since that is the element we want to multiply by the last element
            # in b.
            
            # Now that we have delayed, add the newest value (which requires
            # no delay)
            ztThis = zt.unsqueeze(2).repeat(1, 1, E, 1, 1) # B x 1 x E x H x N
            Sz = torch.cat((ztThis, Sz), dim = 1) # B x (t+1) x E x H x N
            
            # Pad all those values that are not there yet (will multiply b
            # by zero)
            zeroRow = torch.zeros((B, K-(t+1), E, H, N), device = Sz.device)
            SzPad = torch.cat((Sz, zeroRow), dim = 1) # B x K x E x H x N
            
            # Reshape and permute to adapt to multiplication with b (happens
            # outside the if)
            bSz = SzPad.permute(0, 4, 2, 1, 3).reshape(B, N, E*K*H)
        else:
            # Now, we have t>=K which means that Sz is of shape
            #   B x K x E x H x N
            # and thus is full, so we need to get rid of the last element in Sz
            # before adding the new element and multiplying by St.
            
            # We can always get rid of the last element by multiplying by a
            # K x (K-1) selection matrix. So we do that (first we need to
            # permute to have the dimensions ready for multiplication)
            Sz = Sz.permute(0, 2, 3, 4, 1) # B x E x H x N x K
            Sz = torch.matmul(Sz, CK) # B x E x H x N x (K-1)
            Sz = Sz.permute(0, 4, 1, 2, 3) # B x (K-1) x E x H x N
            
            # Get the current time instant
            St = torch.index_select(S, 1, torch.tensor(t, device = S.device))
            #   B x 1 x E x N x N
            # We need to multiply this time instant by all the elements in Sz
            # now, and there are K-1 of those
            St = St.repeat(1, K-1, 1, 1, 1) # B x (K-1) x E x N x N
            # Multiply by the newly acquired St to do one more delay
            Sz = torch.matmul(Sz, St) # B x (K-1) x E x H x N
            
            # Now that we have delayed, add the newest value (which requires
            # no delay)
            ztThis = zt.unsqueeze(2).repeat(1, 1, E, 1, 1) # B x 1 x E x H x N
            Sz = torch.cat((ztThis, Sz), dim = 1) # B x K x E x H x N
            
            # Reshape and permute to adapt to multiplication with b (happens
            # outside the if)
            bSz = Sz.permute(0, 4, 2, 1, 3).reshape(B, N, E*K*H)
        
        # Get back to proper order
        bSz = bSz.permute(0, 2, 1) # B x EKH x N
        #   And multiply with the coefficients
        Bzt = torch.matmul(b, bSz) # B x H x N
        # Now that we have the Bz for this time instant, add the bias
        if zBias is not None:
            Bzt = Bzt + zBias
        # Get the corresponding value of Ax
        Axt = torch.index_select(Ax, 1, torch.tensor(t, device = Ax.device))
        Axt = Axt.reshape(B, H, N)
        # Sum and apply the nonlinearity
        zt = sigma(Axt + Bzt).unsqueeze(1) # B x 1 x H x N
        z = torch.cat((z, zt), dim = 1) # B x (t+1) x H x N
    
    return z # B x T x H x N

#############################################################################
#                                                                           #
#                     LAYERS (Activation Functions)                         #
#                                                                           #
#############################################################################

class MaxLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/15
    """
    MaxLocalActivation creates a localized activation function layer on graphs

    Initialization:

        MaxLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized max activation function layer

    Add graph shift operator:

        MaxLocalActivation.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MaxLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
        # Initialize parameters
        self.reset_parameters()
        
    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        # Change tensor S to numpy now that we have saved it as tensor in self.S
        S = S.cpu().numpy()
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        maxNeighborhoodSizes = []
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = gt.computeNeighborhood(S, k,
                                                            outputType='matrix')
            # compute the k-hop neighborhood
            neighborhood.append(torch.tensor(thisNeighborhood).to(self.S.device))
            maxNeighborhoodSizes.append(thisNeighborhood.shape[1])
        self.maxNeighborhoodSizes = maxNeighborhoodSizes
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's it is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        x = x.unsqueeze(3) # B x F x N x 1
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        for k in range(1,self.K+1):
            x_aux = x.repeat([1, 1, 1, self.maxNeighborhoodSizes[k-1]])
            gatherNeighbor = self.neighborhood[k-1].reshape(
                                                [1,
                                                 1,
                                                 self.N,
                                                 self.maxNeighborhoodSizes[k-1]]
                                                )
            gatherNeighbor = gatherNeighbor.repeat([batchSize, 
                                                    dimNodeSignals,
                                                    1,
                                                    1])
            # And finally we're in position of getting all the neighbors in line
            xNeighbors=torch.gather(x_aux,2,gatherNeighbor.long().to(x.device))
            #   B x F x nOutput x maxNeighbor
            # Note that this gather function already reduces the dimension to
            # nOutputNodes.
            # And proceed to compute the maximum along this dimension
            v, _ = torch.max(xNeighbors, dim = 3)
            v = v.unsqueeze(3) # to concatenate with xK
            xK = torch.cat((xK,v),3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # multiply each k-hop max by corresponding weight
        out = out.reshape([batchSize,dimNodeSignals,self.N])
        return out
    
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.weight.data.uniform_(-stdv, stdv)
    
    def extra_repr(self):
        if self.neighborhood is not None:
            reprString = "neighborhood stored"
        else:
            reprString = "NO neighborhood stored"
        return reprString
    
class MedianLocalActivation(nn.Module):
    # Luana R. Ruiz, rubruiz@seas.upenn.edu, 2019/03/27
    """
    MedianLocalActivation creates a localized activation function layer on 
    graphs

    Initialization:

        MedianLocalActivation(K)

        Inputs:
            K (int): number of hops (>0)

        Output:
            torch.nn.Module for a localized median activation function layer

    Add graph shift operator:

        MedianLocalActivation.addGSO(GSO) Before applying the filter, we need 
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number 
        of edge features is the same; but the number of nodes can change).
        This function also calculates the 0-,1-,...,K-hop neighborhoods of every
        node

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = MedianLocalActivation(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes

        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """

    def __init__(self, K):

        super().__init__()
        assert K > 0 # range has to be greater than 0
        self.K = K
        self.S = None # no GSO assigned yet
        self.N = None # no GSO assigned yet (N learned from the GSO)
        self.neighborhood = 'None' # no neighborhoods calculated yet
        self.masks = 'None' # no mask yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(1,self.K+1))
        # Initialize parameters
        self.reset_parameters()
        
    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        # Change tensor S to numpy now that we have saved it as tensor in self.S
        S = S.cpu().numpy()
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        neighborhood = []
        for k in range(1,self.K+1):
            # For each hop (0,1,...) in the range K
            thisNeighborhood = gt.computeNeighborhood(S, k,
                                                              outputType='list')
            # compute the k-hop neighborhood
            neighborhood.append(thisNeighborhood)
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x N
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.N
        xK = x # xK is a tensor aggregating the 0-hop (x), 1-hop, ..., K-hop
        # max's
        # It is initialized with the 0-hop neigh. (x itself)
        xK = xK.unsqueeze(3) # extra dimension added for concatenation ahead
        #x = x.unsqueeze(3) # B x F x N x 1
        for k in range(1,self.K+1):
            kHopNeighborhood = self.neighborhood[k-1] 
            # Fetching k-hop neighborhoods of all nodes
            kHopMedian = torch.empty(0).to(x.device)
            # Initializing the vector that will contain the k-hop median for
            # every node
            for n in range(self.N):
                # Iterating over the nodes
                # This step is necessary because here the neighborhoods are
                # lists of lists. It is impossible to pad them and feed them as
                # a matrix, as this would impact the outcome of the median
                # operation
                nodeNeighborhood = torch.tensor(np.array(kHopNeighborhood[n]))
                neighborhoodLen = len(nodeNeighborhood)
                gatherNode = nodeNeighborhood.reshape([1, 1, neighborhoodLen])
                gatherNode = gatherNode.repeat([batchSize, dimNodeSignals, 1])
                # Reshaping the node neighborhood for the gather operation
                xNodeNeighbors=torch.gather(x,2,gatherNode.long().to(x.device))
                # Gathering signal values in the node neighborhood
                nodeMedian,_ = torch.median(xNodeNeighbors, dim = 2,
                                            keepdim=True)
                # Computing the median in the neighborhood
                kHopMedian = torch.cat([kHopMedian,nodeMedian],2)
                # Concatenating k-hop medians node by node
            kHopMedian = kHopMedian.unsqueeze(3) # Extra dimension for
            # concatenation with the previous (k-1)-hop median tensor 
            xK = torch.cat([xK,kHopMedian],3)
        out = torch.matmul(xK,self.weight.unsqueeze(2))
        # Multiplying each k-hop median by corresponding trainable weight
        out = out.reshape([batchSize,dimNodeSignals,self.N])
        return out
    
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.K)
        self.weight.data.uniform_(-stdv, stdv)
    
    def extra_repr(self):
        if self.neighborhood is not None:
            reprString = "neighborhood stored"
        else:
            reprString = "NO neighborhood stored"
        return reprString
        
class NoActivation(nn.Module):
    """
    NoActivation creates an activation layer that does nothing
        It is for completeness, to be able to switch between linear models
        and nonlinear models, without altering the entire architecture model
    Initialization:
        NoActivation()
        Output:
            torch.nn.Module for an empty activation layer
    Forward call:
        y = NoActivation(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x number_nodes
        Outputs:
            y (torch.tensor): activated data; shape:
                batch_size x dim_features x number_nodes
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        
        return x
    
    def extra_repr(self):
        reprString = "No Activation Function"
        return reprString
    
#############################################################################
#                                                                           #
#                            LAYERS (Pooling)                               #
#                                                                           #
#############################################################################

class NoPool(nn.Module):
    """
    This is a pooling layer that actually does no pooling. It has the same input
    structure and methods of MaxPoolLocal() for consistency. Basically, this
    allows us to change from pooling to no pooling without necessarily creating
    a new architecture.
    
    In any case, we're pretty sure this function should never ship, and pooling
    can be avoided directly when defining the architecture.
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, GSO):
        # This is necessary to keep the form of the other pooling strategies
        # within the SelectionGNN framework. But we do not care about any GSO.
        pass

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And do not do anything
        return x

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
                self.nInputNodes, self.nOutputNodes, self.nHops)
        reprString += "no neighborhood needed"
        return reprString

class MaxPoolLocal(nn.Module):
    """
    MaxPoolLocal Creates a pooling layer on graphs by selecting nodes

    Initialization:

        MaxPoolLocal(in_dim, out_dim, number_hops)

        Inputs:
            in_dim (int): number of nodes at the input
            out_dim (int): number of nodes at the output
            number_hops (int): number of hops to pool information

        Output:
            torch.nn.Module for a local max-pooling layer.

        Observation: The selected nodes for the output are always the top ones.

    Add a neighborhood set:
        
    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before being used, we need to define the GSO 
        that will determine the neighborhood that we are going to pool.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        v = MaxPoolLocal(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x dim_features x in_dim

        Outputs:
            y (torch.tensor): pooled data; shape:
                batch_size x dim_features x out_dim
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N (And I don't care about E, because the
        # computeNeighborhood function takes care of it)
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        # Get the device (before operating with S and losing it, it's cheaper
        # to store the device now, than to duplicate S -i.e. keep a numpy and a
        # tensor copy of S)
        device = S.device
        # Move the GSO to cpu and to np.array so it can be handled by the
        # computeNeighborhood function
        S = np.array(S.cpu())
        # Compute neighborhood
        neighborhood = gt.computeNeighborhood(S, self.nHops,
                                                      self.nOutputNodes,
                                                      self.nInputNodes,'matrix')
        # And move the neighborhood back to a tensor
        neighborhood = torch.tensor(neighborhood).to(device)
        # The neighborhood matrix has to be a tensor of shape
        #   nOutputNodes x maxNeighborhoodSize
        assert neighborhood.shape[0] == self.nOutputNodes
        assert neighborhood.max() <= self.nInputNodes
        # Store all the relevant information
        self.maxNeighborhoodSize = neighborhood.shape[1]
        self.neighborhood = neighborhood

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        batchSize = x.shape[0]
        dimNodeSignals = x.shape[1]
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And given that the self.neighborhood is already a torch.tensor matrix
        # we can just go ahead and get it.
        # So, x is of shape B x F x N. But we need it to be of shape
        # B x F x N x maxNeighbor. Why? Well, because we need to compute the
        # maximum between the value of each node and those of its neighbors.
        # And we do this by applying a torch.max across the rows (dim = 3) so
        # that we end up again with a B x F x N, but having computed the max.
        # How to fill those extra dimensions? Well, what we have is neighborhood
        # matrix, and we are going to use torch.gather to bring the right
        # values (torch.index_select, while more straightforward, only works
        # along a single dimension).
        # Each row of the matrix neighborhood determines all the neighbors of
        # each node: the first row contains all the neighbors of the first node,
        # etc.
        # The values of the signal at those nodes are contained in the dim = 2
        # of x. So, just for now, let's ignore the batch and feature dimensions
        # and imagine we have a column vector: N x 1. We have to pick some of
        # the elements of this vector and line them up alongside each row
        # so that then we can compute the maximum along these rows.
        # When we torch.gather along dimension 0, we are selecting which row to
        # pick according to each column. Thus, if we have that the first row
        # of the neighborhood matrix is [1, 2, 0] means that we want to pick
        # the value at row 1 of x, at row 2 of x in the next column, and at row
        # 0 of the last column. For these values to be the appropriate ones, we
        # have to repeat x as columns to build our b x F x N x maxNeighbor
        # matrix.
        x = x.unsqueeze(3) # B x F x N x 1
        x = x.repeat([1, 1, 1, self.maxNeighborhoodSize]) # BxFxNxmaxNeighbor
        # And the neighbors that we need to gather are the same across the batch
        # and feature dimensions, so we need to repeat the matrix along those
        # dimensions
        gatherNeighbor = self.neighborhood.reshape([1, 1,
                                                    self.nOutputNodes,
                                                    self.maxNeighborhoodSize])
        gatherNeighbor = gatherNeighbor.repeat([batchSize, dimNodeSignals, 1,1])
        # And finally we're in position of getting all the neighbors in line
        xNeighbors = torch.gather(x, 2, gatherNeighbor)
        #   B x F x nOutput x maxNeighbor
        # Note that this gather function already reduces the dimension to
        # nOutputNodes.
        # And proceed to compute the maximum along this dimension
        v, _ = torch.max(xNeighbors, dim = 3)
        return v

    def extra_repr(self):
        reprString = "in_dim=%d, out_dim=%d, number_hops = %d, " % (
                self.nInputNodes, self.nOutputNodes, self.nHops)
        if self.neighborhood is not None:
            reprString += "neighborhood stored"
        else:
            reprString += "NO neighborhood stored"
        return reprString
    
#############################################################################
#                                                                           #
#                           LAYERS (Filtering)                              #
#                                                                           #
#############################################################################

class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class SpectralGF(nn.Module):
    """
    SpectralGF Creates a (linear) layer that applies a LSI graph filter in the
        spectral domain using a cubic spline if needed.

    Initialization:

        GraphFilter(in_features, out_features, filter_coeff,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_coeff (int): number of filter spectral coefficients
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer) implemented in the spectral domain.

        Observation: Filter taps have shape
            out_features x edge_features x in_features x filter_coeff

    Add graph shift operator:

        SpectralGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = SpectralGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, M, E = 1, bias = True):
        # GSOs will be added later.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.M = M
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, G, M))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.M)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has to have 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S # Save S
        # Now we need to compute the eigendecomposition and save it
        # To compute the eigendecomposition, we use numpy.
        # So, first, get S in numpy format.
        Snp = np.array(S.data.cpu())
        # We will compute the eigendecomposition for each edge feature, so we
        # create the E x N x N space for V, VH and Lambda (we need lambda for
        # the spline kernel)
        V = np.zeros([self.E, self.N, self.N])
        VH = np.zeros([self.E, self.N, self.N])
        Lambda = np.zeros([self.E, self.N])
        # Here we save the resulting spline kernel matrix
        splineKernel = np.zeros([self.E, self.N, self.M])
        for e in range(self.E):
            # Compute the eigendecomposition
            Lambda[e,:], V[e,:,:] = np.linalg.eig(Snp[e,:,:])
            # Compute the hermitian
            VH[e,:,:] = V[e,:,:].conj().T
            # Compute the splineKernel basis matrix
            splineKernel[e,:,:] = gt.splineBasis(self.M, Lambda[e,:])
        # Transform everything to tensors of appropriate type on appropriate
        # device, and store them.
        self.V = torch.tensor(V).type(S.dtype).to(S.device) # E x N x N
        self.VH = torch.tensor(VH).type(S.dtype).to(S.device) # E x N x N
        self.splineKernel = torch.tensor(splineKernel)\
                                .type(S.dtype).to(S.device)
            # E x N x M
        # Once we have computed the splineKernel, we do not need to save the
        # eigenvalues.

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]

        # Check if we have enough spectral filter coefficients as needed, or if
        # we need to fill out the rest using the spline kernel.
        if self.M == self.N:
            self.h = self.weight # F x E x G x N (because N = M)
        else:
            # Adjust dimensions for proper algebraic matrix multiplication
            splineKernel = self.splineKernel.reshape([1,self.E,self.N,self.M])
            # We will multiply a 1 x E x N x M matrix with a F x E x M x G
            # matrix to get the proper F x E x N x G coefficients
            self.h = torch.matmul(splineKernel, self.weight.permute(0,1,3,2))
            # And now we rearrange it to the same shape that the function takes
            self.h = self.h.permute(0,1,3,2) # F x E x G x N
        # And now we add the zero padding (if this comes from a pooling
        # operation)
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output
        u = spectralGF(self.h, self.V, self.VH, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class NodeVariantGF(nn.Module):
    """
    NodeVariantGF Creates a filtering layer that applies a node-variant graph
        filter

    Initialization:

        NodeVariantGF(in_features, out_features, shift_taps, node_taps
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            shift_taps (int): number of filter taps for shifts
            node_taps (int): number of filter taps for nodes
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer using node-variant graph
                filters.

        Observation: Filter taps have shape
            out_features x edge_features x shift_taps x in_features x node_taps

    Add graph shift operator:

        NodeVariantGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = NodeVariantGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, M, E = 1, bias = True):
        # G: Number of input features
        # F: Number of output features
        # K: Number of filter shift taps
        # M: Number of filter node taps
        # GSOs will be added later.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.M = M
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G, M))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K * self.M)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S
        npS = np.array(S.data.cpu()) # Save the GSO as a numpy array because we
        # are going to compute the neighbors.
        # And now we have to fill up the parameter vector, from M to N
        if self.M < self.N:
            # The first elements of M (ordered with whatever order we want)
            # are the ones associated to independent node taps.
            copyNodes = [m for m in range(self.M)]
            # The rest of the nodes will copy one of these M node taps.
            # The way we do this is: if they are connected to one of the M
            # indepdendent nodes, just copy it. If they are not connected,
            # look at the neighbors, neighbors, and so on, until we reach one
            # of the independent nodes.
            # Ties are broken by selecting the node with the smallest index
            # (which, due to the ordering, is the most important node of all
            # the available ones)
            neighborList = gt.computeNeighborhood(npS, 1,
                                                          nb = self.M)
            # This gets the list of 1-hop neighbors for all nodes.
            # Find the nodes that have no neighbors
            nodesWithNoNeighbors = [n for n in range(self.N) \
                                                   if len(neighborList[n]) == 0]
            # If there are still nodes that didn't find a neighbor
            K = 1 # K-hop neighbor we have looked so far
            while len(nodesWithNoNeighbors) > 0:
                # Looks for the next hop
                K += 1
                # Get the neigbors one further hop away
                thisNeighborList = gt.computeNeighborhood(npS,
                                                                  K,
                                                                  nb = self.M)
                # Check if we now have neighbors for those that didn't have
                # before
                for n in nodesWithNoNeighbors:
                    # Get the neighbors of the node
                    thisNodeList = thisNeighborList[n]
                    # If there are neighbors
                    if len(thisNodeList) > 0:
                        # Add them to the list
                        neighborList[n] = thisNodeList
                # Recheck if all nodes have non-empty neighbors
                nodesWithNoNeighbors = [n for n in range(self.N) \
                                                   if len(neighborList[n]) == 0]
            # Now we have obtained the list of independent nodes connected to
            # all nodes, we keep the one with highest score. And since the
            # matrix is already properly ordered, this means keeping the
            # smallest index in the neighborList.
            for m in range(self.M, self.N):
                copyNodes.append(min(neighborList[m]))
            # And, finally create the indices of nodes to copy
            self.copyNodes = torch.tensor(copyNodes).to(S.device)
        elif self.M == self.N:
            # In this case, all parameters go into the vector h
            self.copyNodes = torch.arange(self.M).to(S.device)
        else:
            # This is the rare case in which self.M < self.N, for example, if
            # we train in a larger network and deploy in a smaller one. Since
            # the matrix is ordered by score, we just keep the first N
            # weights
            self.copyNodes = torch.arange(self.N).to(S.device)
        # OBS.: self.weight is updated on each training step, so we cannot
        # define the self.h vector (i.e. the vector with N elements) here,
        # because otherwise it wouldn't be updated every time. So we need, in
        # the for, to use index_select on the actual weights, to create the
        # vector h that is later feed into the NVGF computation.

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # If we have less filter coefficients than the required ones, we need
        # to use the copying scheme
        if self.M == self.N:
            self.h = self.weight
        else:
            self.h = torch.index_select(self.weight, 4, self.copyNodes)
        # And now we add the zero padding
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output
        u = NVGF(self.h, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "shift_taps=%d, node_taps=%d, " % (
                        self.K, self.M) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class EdgeVariantGF(nn.Module):
    """
    EdgeVariantGF Creates a (linear) layer that applies an edge-variant graph
        filter using the masking approach. If less nodes than the total number 
        of nodes are selected, then the remaining nodes adopt an LSI filter
        (i.e. it becomes a hybrid edge-variant grpah filter)

    Initialization:

        EdgeVariantGF(in_features, out_features, shift_taps,
                      selected_nodes, number_nodes,
                      edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            shift_taps (int): number of shifts to consider
            selected_nodes (int): number of selected nodes to implement the EV
                part of the filter
            number_nodes (int): number of nodes
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer using hybrid
                edge-variant graph filters.

        Observation: Filter taps have shape
            out_features x edge_features x shift_taps x in_features
                x number_nodes x number_nodes
            These weights are masked by the corresponding sparsity pattern of
            the graph and the desired number of selected nodes, so only weights
            in the nonzero edges of these nodes will be trained, the
            rest of the parameters contain trash. Therefore, the number of
            parameters will not reflect the actual number of parameters being
            trained.

    Add graph shift operator:

        EdgeVariantGF.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = EdgeVariantGF(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """
    def __init__(self, G, F, K, M, N, E=1, bias = True):
        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.M = M # Number of selected nodes
        self.N = N # Total number of nodes
        self.S = None
        # Create parameters for the Edge-Variant part:
        self.weightEV = nn.parameter.Parameter(torch.Tensor(F, E, K, G, N, N))
        # If we want a hybrid, create parameters
        if self.M < self.N:
            self.weightLSI = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        else:
            self.register_parameter('weightLSI', None)
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K * self.N)
        self.weightEV.data.uniform_(-stdv, stdv)
        if self.weightLSI is not None:
            self.weightLSI.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S # Save the GSO
        # Get the identity matrix across all edge features
        multipleIdentity = torch.eye(self.N).reshape([1, self.N, self.N])\
                            .repeat(self.E, 1, 1).to(S.device)
        # Compute the nonzero elements of S+I_{N}
        sparsityPattern = ((torch.abs(S) + multipleIdentity) > zeroTolerance)
        # Change from byte tensors to float tensors (or the same type of data as
        # the GSO)
        sparsityPattern = sparsityPattern.type(S.dtype)
        # But now we need to kill everything that is between elements M and N
        # (only if M < N)
        if self.M < self.N:
            # Create the ones in the row
            hybridMaskOnesRows = torch.ones([self.M, self.N])
            # Create the ones int he columns
            hybridMaskOnesCols = torch.ones([self.N - self.M, self.M])
            # Create the zeros
            hybridMaskZeros = torch.zeros([self.N - self.M, self.N - self.M])
            # Concatenate the columns
            hybridMask = torch.cat((hybridMaskOnesCols,hybridMaskZeros), dim=1)
            # Concatenate the rows
            hybridMask = torch.cat((hybridMaskOnesRows,hybridMask), dim=0)
        else:
            hybridMask = torch.ones([self.N, self.N])
        # Now that we have the hybrid mask, we need to mask the sparsityPattern
        # we got so far
        hybridMask = hybridMask.reshape([1, self.N, self.N]).to(S.device)
        #   1 x N x N
        sparsityPattern = sparsityPattern * hybridMask
        self.sparsityPattern = sparsityPattern.to(S.device)
        #   E x N x N
        # This gives the sparsity pattern for each edge feature
        # Now, let's create it of the right shape, so we do not have to go
        # around wasting time with reshapes when called in the forward
        # The weights have shape F x E x K x G x N x N
        # The sparsity pattern has shape E x N x N. And we want to make it
        # 1 x E x K x 1 x N x N. The K dimension is to guarantee that for k=0
        # we have the identity
        multipleIdentity = (multipleIdentity * hybridMask)\
                                    .reshape([1, self.E, 1, 1, self.N, self.N])
        if self.K > 1:
            # This gives a 1 x E x 1 x 1 x N x N identity matrix
            sparsityPattern = sparsityPattern\
                                        .reshape([1, self.E, 1, 1, self.N, self.N])
            # This gives a 1 x E x 1 x 1 x N x N sparsity pattern matrix
            sparsityPattern = sparsityPattern.repeat(1, 1, self.K-1, 1, 1, 1)
            # This repeats the sparsity pattern K-1 times giving a matrix of shape
            #   1 x E x (K-1) x 1 x N x N
            sparsityPattern = torch.cat((multipleIdentity,sparsityPattern), dim = 2)
        else:
            sparsityPattern = multipleIdentity
        # This sholud give me a 1 x E x K x 1 x N x N matrix with the identity
        # in the first element
        self.sparsityPatternFull = sparsityPattern.type(S.dtype).to(S.device)

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # Mask the parameters
        self.Phi = self.weightEV * self.sparsityPatternFull
        # And now we add the zero padding
        if Nin < self.N:
            zeroPad = torch.zeros(B, F, self.N-Nin).type(x.dtype).to(x.device)
            x = torch.cat((x, zeroPad), dim = 2)
        # Compute the filter output for the EV part
        uEV = EVGF(self.Phi, x, self.bias)
        # Check if we need an LSI part
        if self.M < self.N:
            # Compute the filter output for the LSI part
            uLSI = LSIGF(self.weightLSI, self.S, x, self.bias)
        else:
            # If we don't, just add zero
            uLSI = torch.tensor(0., dtype = uEV.dtype).to(uEV.device)
        # Add both
        u = uEV + uLSI
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "shift_taps=%d, " % (
                        self.K) + \
                        "selected_nodes=%d, " % (self.M) +\
                        "number_nodes=%d, " % (self.N) +\
                        "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class GraphFilterARMA(nn.Module):
    """
    GraphFilterARMA Creates a (linear) layer that applies a ARMA graph filter
        using Jacobi's method

    Initialization:

        GraphFilterARMA(in_features, out_features,
                        denominator_taps, residue_taps
                        edge_features=1, bias=True, tMax = 5)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            denominator_taps (int): number of filter taps in the denominator
                polynomial
            residue_taps (int): number of filter taps in the residue polynomial
            edge_features (int): number of features over each edge (default: 1)
            bias (bool): add bias vector (one bias per feature) after graph
                filtering (default: True)
            tMax (int): maximum number of Jacobi iterations (default: 5)

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilterARMA.addGSO(GSO) Before applying the filter, we need to 
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterARMA(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, P, K, E = 1, bias = True, tMax = 5):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.P = P
        self.K = K
        self.E = E
        self.tMax = tMax
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.inverseWeight = nn.parameter.Parameter(torch.Tensor(F, E, P, G))
        self.directWeight = nn.parameter.Parameter(torch.Tensor(F, E, P, G))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.inverseWeight.data.uniform_(1.+1./stdv, 1.+2./stdv)
        self.directWeight.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = jARMA(self.inverseWeight, self.directWeight, self.filterWeight,
                  self.S, x, b = self.bias, tMax = self.tMax)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "denominator_taps=%d, " % self.P
        reprString += "residue_taps=%d, " % self.K
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString

class GraphAttentional(nn.Module):
    """
    GraphAttentional Creates a graph attentional layer

    Initialization:

        GraphAttentional(in_features, out_features, attention_heads,
                         edge_features=1, nonlinearity=nn.functional.relu,
                         concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph attentional layer.

    Add graph shift operator:

        GraphAttentional.addGSO(GSO) Before applying the filter, we need to
        define the GSO that we are going to use. This allows to change the GSO
        while using the same filtering coefficients (as long as the number of
        edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1,
        nonlinearity = nn.functional.relu, concatenate = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(K, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(K, E, F, G))
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output
        y = graphAttention(x, self.mixer, self.weight, self.S)
        # This output is of size B x K x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x KF x N such that first iterates over f
            # and then over k: (k=0,f=0), (k=0,f=1), ..., (k=0,f=F-1), (k=1,f=0),
            # (k=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.K*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "attention_heads=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString


class GraphFilterAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads, 
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1, bias = True,
        nonlinearity = nn.functional.relu, concatenate = True):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(P, E, F, G))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(E, K))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output
        y = graphAttentionLSIGF(self.filterWeight, x, self.mixer, self.weight,
                                self.S, b = self.bias)
        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
            # (p=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString

class GraphFilterBatchAttentional_Origin(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1, bias = True,
        nonlinearity = nn.functional.relu, concatenate = True, attentionMode = 'GAT_origin'):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.aij = None
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        self.attentionMode = attentionMode

        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(P, E, F, G))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(E, K))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def returnAttentionGSO(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # Every S has 4 dimensions.
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output

        y, aij = graphAttentionLSIGFBatch_Origin(self.filterWeight, x, self.mixer, self.weight, self.S, b=self.bias)

        self.aij = aij.detach().cpu().numpy()
        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
            # (p=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        reprString += "attentionMode=%s, " % (self.attentionMode)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString

class GraphFilterBatchAttentional_DualHead(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1, bias = True,
        nonlinearity = nn.functional.relu, concatenate = True, attentionMode = 'GAT_DualHead'):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.aij = None
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        self.attentionMode = attentionMode

        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(2*P, E, 2*F))
        self.weight = nn.parameter.Parameter(torch.Tensor(2*P, E, F, G))
        self.weight_bias = nn.parameter.Parameter(torch.Tensor(2*P, E, F))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(2*P, F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def returnAttentionGSO(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # Every S has 4 dimensions.
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output

        y, aij = graphAttentionLSIGFBatch_DualHead(self.filterWeight, x, self.mixer, self.weight, self.weight_bias, self.S, b=self.bias)
        self.aij = aij.detach().cpu().numpy()

        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)

        # When we concatenate we first apply the nonlinearity
        y = self.nonlinearity(y)
        # Concatenate: Make it B x 2*PF x N such that first iterates over f
        # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
        # (p=1,f=1), ..., etc.
        y = y.permute(0, 3, 1, 2)\
                .reshape([B, self.N, 2*self.P*self.F])\
                .permute(0, 2, 1)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        reprString += "attentionMode=%s, " % (self.attentionMode)

        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString


class GraphFilterBatchAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1, bias = True,
        nonlinearity = nn.functional.relu, concatenate = True, attentionMode = 'GAT_modified'):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.aij = None
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        self.attentionMode = attentionMode
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))

        self.weight_bias = nn.parameter.Parameter(torch.Tensor(P, E, F))
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(P, F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        if 'GAT_modified' in attentionMode:
            # Graph Attention Network
            # https://arxiv.org/abs/1710.10903
            self.weight = nn.parameter.Parameter(torch.Tensor(P, E, F, G))
            # print("This is GAT_modified")
            self.graphAttentionLSIGFBatch = graphAttentionLSIGFBatch_modified
        elif attentionMode == 'KeyQuery':
            # Key and Query mode
            # https://arxiv.org/abs/2003.09575
            self.weight = nn.parameter.Parameter(torch.Tensor(P, E, G, G))
            # print("This is KeyQuery")
            self.graphAttentionLSIGFBatch = graphAttentionLSIGFBatch_KeyQuery

        # print("{} Go wrong".format(attentionMode))
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def returnAttentionGSO(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # Every S has 4 dimensions.
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output

        y, aij  = self.graphAttentionLSIGFBatch(self.filterWeight, x, self.mixer, self.weight, self.weight_bias, self.S, b=self.bias)
        self.aij = aij.detach().cpu().numpy()

        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
            # (p=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        reprString += "attentionMode=%s, " % (self.attentionMode)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString




class GraphFilterBatchSimilarityAttentional(nn.Module):
    """
    GraphFilterAttentional Creates a graph convolution attentional layer

    Initialization:

        GraphFilterAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias=True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps (power of the GSO)
            attention_heads (int): number of attention_heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias in the LSIGF stage (default: True)
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for a graph convolution attentional layer.

    Add graph shift operator:

        GraphFilterAttentional.addGSO(GSO) Before applying the filter, we need
        to define the GSO that we are going to use. This allows to change the
        GSO while using the same filtering coefficients (as long as the number
        of edge features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E = 1, bias = True,
        nonlinearity = nn.functional.relu, concatenate = True, attentionMode = 'GAT_Similarity'):
        # P: Number of heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in_features
        self.F = F # out_features
        self.K = K # filter_taps
        self.P = P # attention_heads
        self.E = E # edge_features
        self.S = None # No GSO assigned yet
        self.aij = None
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        self.attentionMode = attentionMode

        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, E, 2*F))
        self.weight_bias = nn.parameter.Parameter(torch.Tensor(P, E, F))
        self.weight = nn.parameter.Parameter(torch.Tensor(P, E, F, G))
        # TODO shall we replace this with modified GAT
        self.filterWeight = nn.parameter.Parameter(torch.Tensor(P, F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.P)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0)
        self.mixer.data.uniform_(-stdv, stdv)
        self.filterWeight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def returnAttentionGSO(self):
        assert len(self.aij.shape) == 5
        # aij is of shape B x P x E x N x N
        assert self.aij.shape[2] == self.E
        self.N = self.aij.shape[3]
        assert self.aij.shape[3] == self.N

        # aij  B x P x E x N x N -> B  x E x N x N
        aij_mean = np.mean(self.aij, axis=1)

        # Every S has 4 dimensions.
        return aij_mean

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # And get the graph attention output
        y, aij = graphSimilarityAttentionLSIGFBatch(self.filterWeight, x, self.mixer, self.weight, self.weight_bias, self.S, b=self.bias)

        self.aij = aij.detach().cpu().numpy()
        # This output is of size B x P x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x PF x N such that first iterates over f
            # and then over p: (p=0,f=0), (p=0,f=1), ..., (p=0,f=F-1), (p=1,f=0),
            # (p=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2)\
                    .reshape([B, self.N, self.P*self.F])\
                    .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim = 1) # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        reprString += "attentionMode=%s, " % (self.attentionMode)

        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString

class EdgeVariantAttentional(nn.Module):
    """
    EdgeVariantAttentional Creates a layer using an edge variant graph filter 
        parameterized by several attention mechanisms

    Initialization:

        EdgeVariantAttentional(in_features, out_features,
                               filter_taps, attention_heads,
                               edge_features=1, bias = True,
                               nonlinearity=nn.functional.relu,
                               concatenate=True)

        Inputs:
            in_features (int): number of input features on top of each node
            out_features (int): number of output features on top of each node
            filter_taps (int): number of filter taps
            attention_heads (int): number of attention heads
            edge_features (int): number of features on top of each edge
                (default: 1)
            bias (bool): include a bias after the EVGF (default: True)    
            nonlinearity (nn.functional): nonlinearity applied after features
                have been updated through attention (default:nn.functional.relu)
            concatenate (bool): If True, the output of the attention_heads
                attention heads are concatenated to form the output features, if
                False, they are averaged (default: True)

        Output:
            torch.nn.Module for an edge variant GF layer, parameterized by the
                attention mechanism.

    Add graph shift operator:

        EdgeVariantAttentional.addGSO(GSO) Before applying the filter, we
        need to define the GSO that we are going to use. This allows to change
        the GSO while using the same filtering coefficients.

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes

    Forward call:

        y = EdgeVariantAttentional(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, P, E=1, bias=True,
                 nonlinearity=nn.functional.relu, concatenate=True):
        # K: Number of filter taps
        # P: Number of attention heads
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G  # in_features
        self.F = F  # out_features
        self.K = K  # filter_taps
        self.P = P  # attention_heads
        self.E = E  # edge_features
        self.S = None  # No GSO assigned yet
        self.nonlinearity = nonlinearity
        self.concatenate = concatenate
        # Create parameters:
        self.mixer = nn.parameter.Parameter(torch.Tensor(P, K, E, 2 * F))
        self.weight = nn.parameter.Parameter(torch.Tensor(P, K, E, F, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        self.mixer.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # And get the graph attention output
        y = graphAttentionEVGF(x, self.mixer, self.weight, self.S, b=self.bias)
        # This output is of size B x K x F x N. Now, we can either concatenate
        # them (inner layers) or average them (outer layer)
        if self.concatenate:
            # When we concatenate we first apply the nonlinearity
            y = self.nonlinearity(y)
            # Concatenate: Make it B x KF x N such that first iterates over f
            # and then over k: (k=0,f=0), (k=0,f=1), ..., (k=0,f=F-1), (k=1,f=0),
            # (k=1,f=1), ..., etc.
            y = y.permute(0, 3, 1, 2) \
                .reshape([B, self.N, self.K * self.F]) \
                .permute(0, 2, 1)
        else:
            # When we don't, we first average
            y = torch.mean(y, dim=1)  # B x F x N
            # And then we apply the nonlinearity
            y = self.nonlinearity(y)

        if Nin < self.N:
            y = torch.index_select(y, 2, torch.arange(Nin).to(y.device))
        return y

    def extra_repr(self):
        reprString = "in_features=%d, " % self.G
        reprString += "out_features=%d, " % self.F
        reprString += "filter_taps=%d, " % self.K
        reprString += "attention_heads=%d, " % self.P
        reprString += "edge_features=%d, " % self.E
        reprString += "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d" % (self.N)
        else:
            reprString += "no GSO stored"
        return reprString


#############################################################################
#                                                                           #
#              LAYERS (Filtering batch-dependent time-varying)              #
#                                                                           #
#############################################################################

class GraphFilter_DB(nn.Module):
    """
    GraphFilter_DB Creates a (linear) layer that applies a graph filter (i.e. a
        graph convolution) considering batches of GSO and the corresponding 
        time delays

    Initialization:

        GraphFilter_DB(in_features, out_features, filter_taps,
                       edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a 
                graph signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter_DB.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                batch_size x time_samples x edge_features 
                                                  x number_nodes x number_nodes

    Forward call:

        y = GraphFilter_DB(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x time_samples x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x time_samples x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x time x dimInFeatures x numberNodesIn
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        # F = x.shape[2]
        assert x.shape[3] == self.N
        # Compute the filter output
        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        # u is of shape batchSize x time x dimOutFeatures x numberNodes
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class HiddenState_DB(nn.Module):
    """
    HiddenState_DB Creates the layer for computing the hidden state of a GRNN

    Initialization:

        HiddenState_DB(signal_features, hidden_features, filter_taps,
                       nonlinearity=torch.tanh, edge_features=1, bias=True)

        Inputs:
            signal_features (int): number of signal features
            hidden_features (int): number of hidden state features
            filter_taps (int): number of filter taps (both filters have the
                same number of filter taps)
            nonlinearity (torch function): nonlinearity applied when computing
                the hidden state
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after each
                filter

        Output:
            torch.nn.Module for a hidden state layer

        Observation: Input-to-Hidden Filter taps have shape
                hidden_features x edge_features x filter_taps x signal_features
            Hidden-to-Hidden FIlter taps have shape
                hidden_features x edge_features x filter_taps x hidden_features

    Add graph shift operator:

    HiddenState_DB.addGSO(GSO) Before applying the layer, we need to define
    the GSO that we are going to use. This allows to change the GSO while
    using the same filtering coefficients (as long as the number of edge
    features is the same; but the number of nodes can change).

    Inputs:
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x time_samples x edge_features 
                                              x number_nodes x number_nodes

    Forward call:

        y = HiddenState_DB(x, z0)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x time_samples x signal_features x number_nodes
            z0 (torch.tensor): initial hidden state; shape:
                batch_size x hidden_features x number_nodes

        Outputs:
            y (torch.tensor): hidden state; shape:
                batch_size x time_samples x hidden_features x number_nodes
    """

    def __init__(self, F, H, K, nonlinearity = torch.tanh, E = 1, bias = True):
        # Initialize parent:
        super().__init__()
        
        # Store the values (using the notation in the paper):
        self.F = F # Input Features
        self.H = H # Hidden Features
        self.K = K # Filter taps
        self.E = E # Number of edge features
        self.S = None
        self.bias = bias # Boolean
        self.sigma = nonlinearity # torch.nn.functional
        
        # Create parameters:
        self.aWeights = nn.parameter.Parameter(torch.Tensor(H, E, K, F))
        self.bWeights = nn.parameter.Parameter(torch.Tensor(H, E, K, H))
        if self.bias:
            self.xBias = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.zBias = nn.parameter.Parameter(torch.Tensor(H, 1))
        else:
            self.register_parameter('xBias', None)
            self.register_parameter('zBias', None)
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.F * self.K)
        self.aWeights.data.uniform_(-stdv, stdv)
        self.bWeights.data.uniform_(-stdv, stdv)
        if self.bias:
            self.xBias.data.uniform_(-stdv, stdv)
            self.zBias.data.uniform_(-stdv, stdv)

    def forward(self, x, z0):
        
        assert self.S is not None

        # Input
        #   S: B x T (x E) x N x N
        #   x: B x T x F x N
        #   z0: B x H x N
        # Output
        #   z: B x T x H x N
        
        # Check dimensions
        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        assert x.shape[2] == self.F
        N = x.shape[3]
        
        assert len(z0.shape) == 3
        assert z0.shape[0] == B
        assert z0.shape[1] == self.H
        assert z0.shape[2] == N
        
        z = GRNN_DB(self.aWeights, self.bWeights,
                    self.S, x, z0, self.sigma,
                    xBias = self.xBias, zBias = self.zBias)
        
        zT = torch.index_select(z, 1, torch.tensor(T-1, device = z.device)) 
        # Give out the last one, to be used as starting point if used in
        # succession
        
        return z, zT.unsqueeze(1)
    
    def addGSO(self, S):
        # Every S has 5 dimensions.
        assert len(S.shape) == 5
        # S is of shape B x T x E x N x N
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S
    
    def extra_repr(self):
        reprString = "in_features=%d, hidden_features=%d, " % (
                        self.F, self.H) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias) +\
                        "nonlinearity=%s" % (self.sigma)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString
    
def matrixPowersBatch(S, K):
    """
    matrixPowers(A_b, K) Computes the matrix powers A_b^k for k = 0, ..., K-1
        for each A_b in b = 1, ..., B.

    Inputs:
        A (tensor): Matrices to compute powers. It can be either a single matrix
            per batch element: shape batch_size x number_nodes x number_nodes
            or contain edge features: shape
                batch_size x edge_features x number_nodes x number_nodes
        K (int): maximum power to be computed (up to K-1)

    Outputs:
        AK: either a collection of K matrices B x K x N x N (if the input was a
            single matrix) or a collection B x E x K x N x N (if the input was a
            collection of E matrices).
    """
    # S can be either a single GSO (N x N) or a collection of GSOs (E x N x N)
    if len(S.shape) == 3:
        B = S.shape[0]
        N = S.shape[1]
        assert S.shape[2] == N
        E = 1
        S = S.unsqueeze(1)
        scalarWeights = True
    elif len(S.shape) == 4:
        B = S.shape[0]
        E = S.shape[1]
        N = S.shape[2]
        assert S.shape[3] == N
        scalarWeights = False

    # Now, let's build the powers of S:
    thisSK = torch.eye(N).repeat([B, E, 1, 1]).to(S.device)
    SK = thisSK.unsqueeze(2)
    for k in range(1, K):
        thisSK = torch.matmul(thisSK, S)
        SK = torch.cat((SK, thisSK.unsqueeze(2)), dim=2)
    # Take out the first dimension if it was a single GSO
    if scalarWeights:
        SK = SK.squeeze(1)

    return SK

def batchLSIGF(h, SK, x, bias=None):
    """
    batchLSIGF(filter_taps, GSO_K, input, bias=None) Computes the output of a
        linear shift-invariant graph filter on input and then adds bias.

    In this case, we consider that there is a separate GSO to be used for each
    of the signals in the batch. In other words, SK[b] is applied when filtering
    x[b] as opposed to applying the same SK to all the graph signals in the
    batch.

    Inputs:
        filter_taps: vector of filter taps; size:
            output_features x edge_features x filter_taps x input_features
        GSO_K: collection of matrices; size:
            batch_size x edge_features x filter_taps x number_nodes x number_nodes
        input: input signal; size:
            batch_size x input_features x number_nodes
        bias: size: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; size:
            batch_size x output_features x number_nodes
    """
    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    B = SK.shape[0]
    assert SK.shape[1] == E
    assert SK.shape[2] == K
    N = SK.shape[3]
    assert SK.shape[4] == N
    assert x.shape[0] == B
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x K x G
    # SK in B x E x K x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N
    SK = SK.permute(1, 2, 0, 3, 4)
    # Now, SK is of shape E x K x B x N x N so that we can multiply by x of
    # size B x G x N to get
    z = torch.matmul(x, SK)
    # which is of size E x K x B x G x N.
    # Now, we have already carried out the multiplication across the dimension
    # of the nodes. Now we need to focus on the K, F, G.
    # Let's start by putting B and N in the front
    z = z.permute(2, 4, 0, 1, 3).reshape([B, N, E * K * G])
    # so that we get z in B x N x EKG.
    # Now adjust the filter taps so they are of the form EKG x F
    h = h.reshape([F, G * E * K]).permute(1, 0)
    # Multiply
    y = torch.matmul(z, h)
    # to get a result of size B x N x F. And permute
    y = y.permute(0, 2, 1)
    # to get it back in the right order: B x F x N.
    # Now, in this case, each element x[b,:,:] has adequately been filtered by
    # the GSO S[b,:,:,:]
    if bias is not None:
        y = y + bias
    return y

class GraphFilterBatchGSO(GraphFilter):
    """
    GraphFilterBatchGSO Creates a (linear) layer that applies a graph filter
        with a different GSO for each signal in the batch.

    This function is typically useful when not only the graph signal is changed
    during training, but also the GSO. That is, each data point in the batch is
    of the form (x_b,S_b) for b = 1,...,B instead of just x_b. The filter
    coefficients are still the same being applied to all graph filters, but both
    the GSO and the graph signal are different for each datapoint in the batch.

    Initialization:

        GraphFilterBatchGSO(in_features, out_features, filter_taps,
                            edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilterBatchGSO.addGSO(GSO) Before applying the filter, we need to
        define the GSOs that we are going to use for each element of the batch.
        Each GSO has to have the same number of edges, but the number of nodes
        can change.

        Inputs:
            GSO (tensor): collection of graph shift operators; size can be
                batch_size x number_nodes x number_nodes, or
                batch_size x edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilterBatchGSO(x)

        Inputs:
            x (tensor): input data; size: batch_size x in_features x number_nodes

        Outputs:
            y (tensor): output; size: batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__(G, F, K, E, bias)

    def addGSO(self, S):
        # So, we have to take into account the situation where S is either
        # B x N x N or B x E x N x N. No matter what, we're always handling,
        # internally the dimension E. So if the input is B x N x N, we have to
        # unsqueeze it so it becomes B x 1 x N x N.
        if len(S.shape) == 3 and S.shape[1] == S.shape[2]:
            self.S = S.unsqueeze(1)
        elif len(S.shape) == 4 and S.shape[1] == self.E \
                and S.shape[2] == S.shape[3]:
            self.S = S
        else:
            # TODO: print error
            pass

        self.N = self.S.shape[2]
        self.B = self.S.shape[0]
        self.SK = matrixPowersBatch(self.S, self.K)

    def forward(self, x):
        # TODO: If S (and consequently SK) hasn't been defined, print an error.
        return batchLSIGF(self.weight, self.SK, x, self.bias)

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d, batch_size=%d" % (
                self.N, self.B)
        else:
            reprString += "no GSO stored"
        return reprString

def BatchLSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.

    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.

    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.

    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            batch_size x edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}

    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[1] == E
    N = S.shape[2]
    assert S.shape[3] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in B x E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in B x E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    # print(S)
    S = S.reshape([B, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S.float()) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

class GraphFilterBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E = 1, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = BatchLSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString





def torchpermul(h, x, b=None):

    # h is output_features x edge_weights x filter_taps x input_features
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:

    # in the notation we've been using:
    # h in G x H
    # x in B x H x N
    # b in G x N
    # y in B x G x N

    # Now, we have x in B x H x N and h in G x H
    # B x N x H with H x G -> B x N x G -> B x G x N
    y = torch.mul(x.permute(0, 2, 1), h.permute(1, 0)).permute(0, 2, 1)  # B x G x N

    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

class GraphFilterMoRNNBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, H, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.H = H  # hidden_features
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight_A = nn.parameter.Parameter(torch.Tensor(H, E, K, G))
        self.weight_B = nn.parameter.Parameter(torch.Tensor(H, H))
        self.weight_D = nn.parameter.Parameter(torch.Tensor(F, H))
        if bias:
            self.bias_A = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_B = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_D = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # todo: check initialize weight
        # Taken from _ConvNd initialization of parameters:
        stdv_a = 1. / math.sqrt(self.G * self.K)
        self.weight_A.data.uniform_(-stdv_a, stdv_a)
        if self.bias_A is not None:
            self.bias_A.data.uniform_(-stdv_a, stdv_a)

        stdv_b = 1. / math.sqrt(self.H)
        self.weight_B.data.uniform_(-stdv_b, stdv_b)
        if self.bias_B is not None:
            self.bias_B.data.uniform_(-stdv_b, stdv_b)

        stdv_d = 1. / math.sqrt(self.H )
        self.weight_D.data.uniform_(-stdv_d, stdv_d)
        if self.bias_D is not None:
            self.bias_D.data.uniform_(-stdv_d, stdv_d)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def updateHiddenState(self, hiddenState):

        self.hiddenState = hiddenState

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u_a = BatchLSIGF(self.weight_A, self.S, x, self.bias_A) # B x H x n
        # u_b = torch.mul(self.hiddenState.permute(0,2,1), self.weight_B.permute(1, 0)).permute(0,2,1) + self.bias_B # B x H x n
        u_b = torchpermul(self.weight_B,self.hiddenState,self.bias_B)
        sigma = nn.ReLU(inplace=True)
        # sigma = nn.Tanh()
        self.hiddenStateNext = sigma(u_a + u_b)

        # v1
        # u = torch.mul(self.weight_D, self.hiddenState) + self.bias_D

        # v2
        # u = torch.mul(u_a.permute(0,2,1), self.weight_D.permute(1, 0)).permute(0,2,1) + self.bias_D
        u = torchpermul(self.weight_D, self.hiddenStateNext, self.bias_D)

        self.updateHiddenState(self.hiddenStateNext)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, hidden_features=%d, " % (
                        self.G, self.F, self.H) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias_D is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString


class GraphFilterL2ShareBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter

    Initialization:

        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)

        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering

        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).

        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features

    Add graph shift operator:

        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).

        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes

    Forward call:

        y = GraphFilter(x)

        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes

        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, H, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.H = H  # hidden_features
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight_A = nn.parameter.Parameter(torch.Tensor(H, E, K, G))
        self.weight_B = nn.parameter.Parameter(torch.Tensor(H, H))
        self.weight_D = nn.parameter.Parameter(torch.Tensor(F, H))
        if bias:
            self.bias_A = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_B = nn.parameter.Parameter(torch.Tensor(H, 1))
            self.bias_D = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # todo: check initialize weight
        # Taken from _ConvNd initialization of parameters:
        stdv_a = 1. / math.sqrt(self.G * self.K)
        self.weight_A.data.uniform_(-stdv_a, stdv_a)
        if self.bias_A is not None:
            self.bias_A.data.uniform_(-stdv_a, stdv_a)

        stdv_b = 1. / math.sqrt(self.H)
        self.weight_B.data.uniform_(-stdv_b, stdv_b)
        if self.bias_B is not None:
            self.bias_B.data.uniform_(-stdv_b, stdv_b)

        stdv_d = 1. / math.sqrt(self.H )
        self.weight_D.data.uniform_(-stdv_d, stdv_d)
        if self.bias_D is not None:
            self.bias_D.data.uniform_(-stdv_d, stdv_d)

    def addGSO(self, S):
        # Every S has 4 dimensions.
        assert len(S.shape) == 4
        # S is of shape B x E x N x N
        assert S.shape[1] == self.E
        self.N = S.shape[2]
        assert S.shape[3] == self.N
        self.S = S

    def updateHiddenState(self, hiddenState):

        self.hiddenState = hiddenState

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u_a = BatchLSIGF(self.weight_A, self.S, x, self.bias_A) # B x H x n
        u_b = torchpermul(self.weight_B, self.hiddenState, self.bias_B)
        
        sigma = nn.ReLU(inplace=True)
        # sigma = nn.Tanh()
        self.hiddenStateNext = sigma(u_a + u_b)


        # u = torch.mul(u_a.permute(0,2,1), self.weight_D.permute(1, 0)).permute(0,2,1) + self.bias_D
        u = torchpermul(self.weight_D, self.hiddenStateNext, self.bias_D)

        self.updateHiddenState(self.hiddenStateNext)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, hidden_features=%d, " % (
                        self.G, self.F, self.H) + "filter_taps=%d, " % (
                        self.K) + "edge_features=%d, " % (self.E) +\
                        "bias=%s, " % (self.bias_D is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString
