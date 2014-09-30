__author__ = 'ank'

import numpy as np
import matplotlib.pyplot as plt
import PIL
import mdp
from itertools import product
from pickle import load, dump
from os import path
from time import time
from skimage.segmentation import random_walker, mark_boundaries
from skimage.morphology import convex_hull_image, label, dilation, disk
from skimage.filter import gaussian_filter
from skimage.measure import perimeter
from matplotlib import colors
from pylab import get_cmap
# from skimage.measure import label
# todo: add the time optimization and management for varying window sizes for Gabor filter

selem = disk(10)
debug = False
timing = True

def rs(matrix, name):
    plt.title(name)
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.clf()

def debug_wrapper(funct):

    def check_matrix(*args,**kwargs):
        result = funct(*args, **kwargs)
        if debug:
            rs(result, funct.__name__)
        check_matrix.__name__ = funct.__name__
        check_matrix.__doc__ = funct.__doc__
        return result

    return check_matrix

def time_wrapper(funct):

    def time_execution(*args,**kwargs):
        start = time()
        result = funct(*args, **kwargs)
        if timing:
            print funct.__name__, time()-start
        time_execution.__doc__ = funct.__doc__
        return result

    return time_execution

@time_wrapper
@debug_wrapper
def import_image(image_to_load):
    col = PIL.Image.open(image_to_load)
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw = bw - np.min(bw)
    bw = bw.astype(np.float64)/float(np.max(bw))
    return bw

@time_wrapper
@debug_wrapper
def import_edited(buffer_directory):
    if path.exists(buffer_directory+'-'+"EDIT_ME2.tif"):
        col = PIL.Image.open(buffer_directory+'-'+"EDIT_ME2.tif")
    else:
        col = PIL.Image.open(buffer_directory+'-'+"EDIT_ME.tif")
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw<=120] = 0
    bw[bw>120] = 1
    return bw

@time_wrapper
@debug_wrapper
def gabor(bw_image, freq, scale, scale_distortion=1., self_cross=False, field=10):

    # gabor filter normalization with respect to the surface convolution
    def check_integral(gabor_filter):
        ones = np.ones(gabor_filter.shape)
        avg = np.average(ones*gabor_filter)
        return gabor_filter-avg

    quality = 16
    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    phis = [pi]
    size = (field, field)
    sgm = (5*scale, 3*scale*scale_distortion)

    nfilters = len(orientations)*len(phis)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, (alpha, phi) in enumerate(product(orientations, phis)):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        if self_cross:
            arr=np.minimum(arr, mdp.utils.gabor(size, alpha+pi/2, phi, freq, sgm))
        arr = check_integral(arr)
        gabors[i, :, :] = arr
    #     plt.subplot(6,6,i+1)
    #     plt.title('%s, %s, %s, %s'%('{0:.2f}'.format(alpha), '{0:.2f}'.format(phi), freq, sgm))
    #     plt.imshow(arr, cmap = 'gray', interpolation='nearest')
    # plt.show()
    # plt.clf()
    node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill', fillvalue=0, output_2d=False)
    cim = node.execute(bw_image[np.newaxis, :, :])
    sum2 = np.zeros(cim[0, 0, :, :].shape)
    for i in range(0, nfilters):
        pr_cim = cim[0, i, :, :]
        sum2 = sum2 - pr_cim
    sum2[sum2>0] = sum2[sum2>0]/np.max(sum2)
    sum2[sum2<0] = -sum2[sum2<0]/np.min(sum2)
    return sum2

@time_wrapper
@debug_wrapper
def cluster_by_diffusion(data):
    markers = np.zeros(data.shape, dtype=np.uint8)
    markers[data < -0.2] = 1
    markers[data > 0.3] = 2
    labels2 = random_walker(data, markers, beta=10, mode='cg_mg')
    return labels2

@time_wrapper
@debug_wrapper
def cluster_process(labels):
    rbase = np.zeros(labels.shape)
    rubase = np.zeros(labels.shape)
    rubase[range(0,20),:] = 1
    rubase[:,range(0,20)] = 1
    rubase[range(-20,-1),:] = 1
    rubase[:,range(-20,-1)] = 1
    for i in range(1, int(np.max(labels))):
        base = np.zeros(labels.shape)
        base[labels==i] = 1
        li = len(base.nonzero()[0])
        if li>0:
            hull = convex_hull_image(base)
            lh =len(hull.nonzero()[0])
            cond = (li>4000 and float(lh)/float(li)<1.07 and perimeter(base)**2.0/li<20) or np.max(base*rubase)>0.5
            if cond:
                rbase = rbase + base
    rbase[rubase.astype(np.bool)] = 1
    return dilation(rbase,selem)


@time_wrapper
def repaint_culsters(clusterNo=100):
    prism_cmap = get_cmap('prism')
    prism_vals = prism_cmap(np.arange(clusterNo))
    prism_vals[0] = [0, 0, 0, 1]
    costum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', prism_vals)
    return costum_cmap

@time_wrapper
def human_loop(buffer_directory, image_to_import):
    start = time()
    bw = import_image(image_to_import)
    # Human
    sum2 = gabor(bw, 1/8., 1, self_cross=True, field=20)

    # The separator is acting here:
    sum20 = gabor(bw, 1/4., 0.5, field=20)
    sum20[sum20>-0.1] = 0
    sum2  = sum2 + sum20
    # Mice part
    sum20 = gabor(bw, 1/10., 1, scale_distortion=1.5, field=20)
    sum20[sum20>-0.1] = 0
    sum2  = sum2 + sum20
    ###########################################

    bw_blur = gaussian_filter(bw, 10)
    bwth = np.zeros(bw_blur.shape)
    bwth[bw_blur>0.3] = 1
    clsts = (label(bwth)+1)*bwth

    rbase = cluster_process(clsts)[9:,:][:,9:][:-10,:][:,:-10]

    rim = PIL.Image.fromarray((rbase*254).astype(np.uint8))
    rim.save(buffer_directory+'-'+"I_AM_UNBROKEN_NUCLEUS.bmp")

    sum22 = np.copy(sum2)
    sum22[sum2<0] = 0
    d_c = cluster_by_diffusion(sum2)
    rebw = bw[9:,:][:,9:][:-10,:][:,:-10]
    reim = PIL.Image.fromarray((rebw/np.max(rebw)*254).astype(np.uint8))
    reim.save(buffer_directory+'-'+"I_AM_THE_ORIGINAL.tif")
    seg_dc = (label(d_c, neighbors=4)+1)*(d_c-1)
    redd = set(seg_dc[rbase>0.01].tolist())
    for i in redd:
        seg_dc[seg_dc==i] = 0
    d_c = d_c*0
    d_c[seg_dc>0] = 1
    int_arr = np.asarray(np.dstack((d_c*254, d_c*254, d_c*0)), dtype=np.uint8)
    msk = PIL.Image.fromarray(int_arr)
    msk.save(buffer_directory+'-'+"EDIT_ME.tif")
    dump(rebw ,open(buffer_directory+'-'+'DO_NOT_TOUCH_ME.dmp','wb'))
    return time()-start

@time_wrapper
def human_afterloop(output_directory, pre_time, fle_name, buffer_directory):
    start2 = time()

    d_c = import_edited(buffer_directory)
    rebw = load(open(buffer_directory+'-'+'DO_NOT_TOUCH_ME.dmp','rb'))
    seg_dc = (label(d_c,neighbors=4)+1)*d_c
    if np.max(seg_dc)<4:
        return 'FAILED: mask for %s looks unsegmented' % fle_name

    colormap = repaint_culsters(int(np.max(seg_dc)))

    segs = len(set(seg_dc.flatten().tolist()))-1

    # shows the result before saving the clustering and printing to the user the number of the images
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(rebw, cmap='gray', interpolation='nearest')

    plt.subplot(1,2,2)
    plt.title('Segmentation - clusters: %s'%str(segs))
    plt.imshow(mark_boundaries(rebw, d_c))
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest', alpha=0.3)

    plt.show()

    plt.imshow(mark_boundaries(rebw, d_c))
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest', alpha=0.3)

    plt.savefig(path.join(output_directory, fle_name+'_%s_clusters.png'%str(segs)), dpi=500, bbox_inches='tight', pad_inches=0.0)

    return fle_name+'\t clusters: %s,\t total time : %s'%(segs, "{0:.2f}".format(time()-start2+pre_time))

if __name__ == "__main__":
    pass