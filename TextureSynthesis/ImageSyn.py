import numpy as np
from scipy.optimize import minimize
from TextureSynthesis.Misc import *
from TextureSynthesis.mask.mask_generate import *

def ImageSyn(net1, net2, constraints, init=None, bounds=None, callback=None, minimize_options=None, gradient_free_region=None):
    if init==None:
        init = np.random.randn(*net1.blobs['data'].data.shape)
    
    layers, indices = get_indices(net1, constraints)
    
    def f(x):
        x = x.reshape(*net1.blobs['data'].data.shape)
        net1.forward(data=x, end=list(layers)[min(len(layers)-1, indices[0]+1)])
        net2.forward(data=x, end=list(layers)[min(len(layers)-1, indices[0]+1)])
        f_val = 0
        #clear gradient in all layers
        for index in indices:
            net1.blobs[list(layers)[index]].diff[...] = np.zeros_like(net1.blobs[list(layers)[index]].diff)
            net2.blobs[list(layers)[index]].diff[...] = np.zeros_like(net2.blobs[list(layers)[index]].diff)    

        for i,index in enumerate(indices):
            layer = list(layers)[index]
            for l,loss_function in enumerate(constraints[layer].loss_functions):
                constraints[layer].parameter_lists[l].update({'activations1': net1.blobs[layer].data.copy()})
                constraints[layer].parameter_lists[l].update({'activations2': net2.blobs[layer].data.copy()})
                val, grad = loss_function(**constraints[layer].parameter_lists[l])
                f_val += val

                #GRADIENT MASK
                mask = gen_mask(grad.shape[2])
                grad_mask = np.zeros((grad.shape))
                for num in range(0, grad.shape[1]):
                    grad_mask[0,num] = mask

                net1.blobs[layer].diff[:] += grad * grad_mask
                net2.blobs[layer].diff[:] += grad * (1 - grad_mask)

            #gradient wrt inactive units is 0
            net1.blobs[layer].diff[(net1.blobs[layer].data == 0)] = 0.
            net2.blobs[layer].diff[(net2.blobs[layer].data == 0)] = 0.
            if index == indices[-1]:
                f_grad1 = net1.backward(start=layer)['data'].copy()
                f_grad2 = net2.backward(start=layer)['data'].copy()
            else:        
                net1.backward(start=layer, end=list(layers)[indices[i+1]])
                net2.backward(start=layer, end=list(layers)[indices[i+1]])
        mask = gen_mask(16)
        mask_sum = np.sum(mask)/(16*16)
        #print("mask_sum is " + str(mask_sum)) 
        f_grad = f_grad1/mask_sum + f_grad2/(1-mask_sum) #this needs to be generalized

        if gradient_free_region!=None:
            f_grad[gradient_free_region==1] = 0    

        return [f_val, np.array(f_grad.ravel(), dtype=float)]            
        
    result = minimize(f, init,
                          method='L-BFGS-B', 
                          jac=True,
                          bounds=bounds,
                          callback=callback,
                          options=minimize_options)
    return result