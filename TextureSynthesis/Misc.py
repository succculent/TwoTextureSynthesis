import numpy as np
import scipy
import caffe
import matplotlib.pyplot as plt
from IPython.display import display,clear_output
import time
from TextureSynthesis.mask.mask_generate import *

class constraint(object):
    def __init__(self, loss_functions, parameter_lists):
        self.loss_functions = loss_functions
        self.parameter_lists = parameter_lists
  
def get_indices(net, constraints):
    '''
    extracts layers and indicies from the network
    '''

    indices = [ndx for ndx,layer in enumerate(net.blobs.keys()) if layer in constraints.keys()]
    return net.blobs.keys(),indices[::-1]

def get_bounds(images1, images2, im_size):
    '''
    Gets the optimization bounds

    :images1: image of first texture loaded into the network using load_image
    :images2: image of second texture loaded into the network using load_image
    :im_size: image side length in pixels
    '''

    lowerbound = min(np.min([im.min() for im in images1]), np.min([im.min() for im in images2]))
    upperbound = max(np.max([im.max() for im in images1]), np.max([im.max() for im in images2]))
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds 

def gram_matrix(activations1, activations2):
    '''
    Calculates the gram matrix according to the mask and two inputs

    :activations1: activation layers from the VGG network for the first input image
    :activations2: activation layers from the VGG network for the second input image
    '''

    mask = gen_mask(activations1.shape[2])
    N = activations1.shape[1]
    new_mask = np.zeros(activations1.shape)
    for i in range(0, N):
        new_mask[0,i] = mask
    a1 = activations1*new_mask
    a2 = activations2*(1-new_mask)
    F1 = a1.reshape(N,-1)
    F2 = a2.reshape(N,-1)
    M = F1.shape[1]
    G = np.dot(F1,F1.T) / M + np.dot(F2,F2.T) / M
    return G

def uniform_hist(X):
    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return matched_image

def load_image(img, im_size, net_model, net_weights, mean):
    '''
    Loads and preprocesses image into caffe format by constructing and using the appropriate network.

    :param img: img pre-loaded using caffe.io.load_image
    :param im_size: size of the image after preprocessing if float that the original image is rescaled to contain im_size**2 pixels
    :param net_model: file name of the prototxt file defining the network model
    :param net_weights: file name of caffemodel file defining the network weights
    :param mean: mean values for each color channel (bgr) which are subtracted during preprocessing
    :return: preprocessed image and caffe.Classifier object defining the network
    '''
    if isinstance(im_size,float):
        im_scale = np.sqrt(im_size**2 /np.prod(np.asarray(img.shape[:2])))
        im_size = im_scale * np.asarray(img.shape[:2])
    batchSize = 1
    with open(net_model,'r+') as f:
        data = f.readlines() 
    data[2] = "input_dim: %i\n" %(batchSize)
    data[4] = "input_dim: %i\n" %(im_size[0])
    data[5] = "input_dim: %i\n" %(im_size[1])
    with open(net_model,'r+') as f:
        f.writelines(data)
    net_mean =  np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
    #load pretrained network
    net = caffe.Classifier( 
    net_model, net_weights,
    mean = net_mean,
    channel_swap=(2,1,0),
    input_scale=255,)
    img_pp = net.transformer.preprocess('data',img)[None,:]
    return[img_pp, net]