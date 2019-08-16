import glob
import sys
import os
from collections import OrderedDict
import caffe
import numpy as np
base_dir = os.getcwd()
sys.path.append(base_dir)
from TextureSynthesis import *
from matplotlib.pyplot import imshow, figure, savefig

VGGweights = os.path.join(base_dir, 'Models/vgg_normalised.caffemodel')
VGGmodel = os.path.join(base_dir, 'Models/VGG_ave_pool_deploy.prototxt')
imagenet_mean = np.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)
im_dir = os.path.join(base_dir, 'Images/')
gpu = 0
caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
caffe.set_device(gpu)

input1_name = 'a.png' #change this according to file name and extension 
input2_name = 'b.png' #change this according to file name and extension 
output_name = 'result_texture' #change according to desired output name

source_img_name = glob.glob1(im_dir, input1_name)[0]
img1 = caffe.io.load_image(im_dir + source_img_name)
source_img_name2 = glob.glob1(im_dir, input2_name)[0]
img2 = caffe.io.load_image(im_dir + source_img_name2)
im_size = 256.

[source_img1, net1] = load_image(img1, im_size, 
                            VGGmodel, VGGweights, imagenet_mean)
[source_img2, net2] = load_image(img2, im_size, 
	                        VGGmodel, VGGweights, imagenet_mean)
im_size = np.asarray(source_img1.shape[-2:])

#l-bfgs parameters optimisation
maxiter = 10
m = 20

#define layers to include in the texture model and weights w_l
tex_layers = ['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1'] #possibly do not need to include all of these for testing
tex_weights = [1e9,1e9,1e9,1e9,1e9] #this comes hand in hand with the one above

#pass image through the network and save the constraints on each layer
constraints = OrderedDict()
net1.forward(data = source_img1)
net2.forward(data = source_img2)
for l,layer in enumerate(tex_layers):
    constraints[layer] = constraint([LossFunctions.gram_mse_loss],
                                    [{'target_gram_matrix': gram_matrix(net1.blobs[layer].data, net2.blobs[layer].data),
                                     'weight': tex_weights[l]}])
	    
#get optimisation bounds
bounds = get_bounds([source_img1],[source_img2],im_size)

#generate new texture
result = ImageSyn(net1, net2, constraints, bounds=bounds, callback=lambda x: show_progress(x,net1),
                  minimize_options={'maxiter': maxiter,
                                    'maxcor': m,
                                    'ftol': 0, 'gtol': 0})

#match histogram of new texture with that of the source texture and show both images
new_texture = result['x'].reshape(*source_img1.shape[1:]).transpose(1,2,0)[:,:,::-1]
new_texture = histogram_matching(new_texture, (img1+img2)/2)
imshow(new_texture)
savefig(output_name + '.png') #can change the output file extention to any available in matplotlib