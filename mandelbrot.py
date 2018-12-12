# -*- coding: utf-8 -*-
# from __future__ import unicode_literals

"""
draw mandelbrot image,
refer to: Mandelbrot 集合，将 TensorFlow 应用于普通数学 http://t.cn/EUtXxaU

Authors: etworker
"""

import os
from os import path as osp
import tensorflow as tf
import numpy as np
import cv2


def calc_mandelbrot(coefficient_list=[1,0,1], step_num=200):
    """
    calculate mandelbrot,
    coefficient_list is coefficeient of xs, z, z^2, ...
    e.g. [1,0,1] means xs + z^2
    """

    sess = tf.InteractiveSession()

    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    Z = X + 1j*Y

    xs = tf.constant(Z.astype(np.complex64))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    tf.global_variables_initializer().run()

    zs_ = coefficient_list[0] * xs
    for i in range(1, len(coefficient_list)):
        zs_ += coefficient_list[i]*np.power(zs, i)
    
    not_diverged = tf.abs(zs_) < 4
    step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))

    for _ in range(step_num): step.run()

    result = ns.eval()
    sess.close()

    return result 


def get_fractal_img(data):
    """
    convert data to image
    """

    cyclic = (6.28*data/20.0).reshape(list(data.shape)+[1])
    img = np.concatenate([
        10+20*np.cos(cyclic),
        30+50*np.sin(cyclic),
        155-80*np.cos(cyclic)], 2)
    img[data==data.max()] = 0
    img = np.uint8(np.clip(img, 0, 255))

    return img
    

def try_combination(bitlen=6):
    """
    try combination, all coefficient of z^k is 0/1
    """

    pic_dir = 'pic'
    if not osp.isdir(pic_dir):
        os.makedirs(pic_dir)

    coefficient_combination = []
    n = pow(2, bitlen)
    while n>0:
        b = '{0:0%db}' % bitlen
        b = list(b.format(n))
        b = list(map(lambda x:int(x), b))
        coefficient_combination.append([1] + b)
        n -= 1
    
    step_num = 100
    for coefficient_list in coefficient_combination:
        fn = 'mandelbrot_%s_%d' % ('-'.join(list(map(lambda x:str(x), coefficient_list))), step_num)
        filename = osp.join(pic_dir, fn+'.jpg')
        print('calc %s ...' % fn)
        if osp.isfile(filename): continue

        data = calc_mandelbrot(coefficient_list, step_num)
        img = get_fractal_img(data)        
        cv2.imwrite(filename, img)

    print('finished, total %d' % len(coefficient_combination))


if __name__ == "__main__":
    try_combination(bitlen=6)