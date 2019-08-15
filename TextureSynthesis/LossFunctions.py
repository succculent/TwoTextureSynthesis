import numpy as np
import pdb
from TextureSynthesis.mask.mask_generate import *

def gram_mse_loss(activations1, activations2, target_gram_matrix, weight=1.):
    N = activations1.shape[1]
    fm_size = np.array(activations1.shape[2:])
    M = np.prod(fm_size)
    G_target = target_gram_matrix

    #fit the mask to the activation layer
    mask = gen_mask(activations1.shape[2])
    new_mask = np.zeros(activations1.shape)
    for i in range(0, N):
        new_mask[0,i] = mask
    F1 = (new_mask*activations1).reshape(N,-1)
    F2 = ((1-new_mask)*activations2).reshape(N,-1)

    #calculate independent gram matricies and add them together
    G1 = np.dot(F1,F1.T) / M
    G2 = np.dot(F2,F2.T) / M
    G = G1 + G2

    #calculate loss and gradient to return
    loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
    gradient = (weight * np.dot(F1.T+F2.T, (G - G_target)).T / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
    return [loss, gradient]