import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import math
import os 

## HELPER
def random_int(shape=[], minval=0, maxval=1):
    return tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)

def random_float(shape=[], minval=0.0, maxval=1.0):
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd

def debuglogger(filename, txt):
    if os.path.exists(filename):
      f = open(filename, "a")
    else:
      f = open(filename,"w")
    f.write(txt+'\n')
    f.close()

def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    #rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
#     c1   = tf.math.cos(rotation)
#     s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
#     rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
#                                    -s1,  c1,   zero, 
#                                    zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                               zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    

    return  K.dot(shear_matrix,K.dot(zoom_matrix, shift_matrix)) #K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))                  

def transform(image, DIM, ROT, SHR, H_ZOOM, V_ZOOM, H_SHIFT, V_SHIFT, FILL_MODE):#[rot,shr,h_zoom,w_zoom,h_shift,w_shift]):
    if DIM[0]>DIM[1]:
        diff  = (DIM[0]-DIM[1])
        pad   = [diff//2, diff//2 + (1 if diff%2 else 0)]
        image = tf.pad(image, [[0, 0], [pad[0], pad[1]],[0, 0]])
        NEW_DIM = DIM[0]
    elif DIM[0]<DIM[1]:
        diff  = (DIM[1]-DIM[0])
        pad   = [diff//2, diff//2 + (1 if diff%2 else 0)]
        image = tf.pad(image, [[pad[0], pad[1]], [0, 0],[0, 0]])
        NEW_DIM = DIM[1]
    
    rot = ROT * tf.random.normal([1], dtype='float32')
    shr = SHR * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / H_ZOOM
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / V_ZOOM
    h_shift = H_SHIFT * tf.random.normal([1], dtype='float32') 
    w_shift = V_SHIFT * tf.random.normal([1], dtype='float32') 
    
    transformation_matrix=tf.linalg.inv(get_mat(shr,h_zoom,w_zoom,h_shift,w_shift))
    
    flat_tensor=tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)
    
    image=tfa.image.transform(image,flat_tensor, fill_mode=FILL_MODE)
    
    rotation = math.pi * rot / 180.
    
    image=tfa.image.rotate(image,-rotation, fill_mode=FILL_MODE)
    
    if DIM[0]>DIM[1]:
        image=tf.reshape(image, [NEW_DIM, NEW_DIM,3])
        image = image[:, pad[0]:-pad[1],:]
    elif DIM[1]>DIM[0]:
        image=tf.reshape(image, [NEW_DIM, NEW_DIM,3])
        image = image[pad[0]:-pad[1],:,:]
    image = tf.reshape(image, [*DIM, 3])    
    return image

def dropout(image, DIM=[360, 640], PROBABILITY = 0.6, CT = 5, SZ = 0.1):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    if (P==0)|(CT==0)|(SZ==0): 
        return image
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM[1]),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM[0]),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast( SZ*min(DIM),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM[0],y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM[1],x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3], dtype = image.dtype) 
        three = image[ya:yb,xb:DIM[1],:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM[0],:,:]],axis=0)
        image = tf.reshape(image,[*DIM,3])
    return image

def RandomBorder(img, target_size, mode, fill_border, remove_symbol):
    height = target_size[0]
    if mode=='random':
        crop_dim = tf.cast(random_float([], 10, 30)/360*height, tf.int32) # max:35
    elif mode=='constant':
        crop_dim = tf.cast(18/360*height, tf.int32) # max:35
    else:
        NotImplemented
        
    new_img = img[:-crop_dim,:,:]
    
    if fill_border:
        constant_values = tf.cast(0, img.dtype) # random_int([], 245, 250)(white) perform poorly
        new_img = tf.pad(new_img, [[0, crop_dim],[0,0],[0,0]], mode='constant',
                         constant_values=constant_values)
    else:
        new_img = tf.image.resize(new_img, tf.shape(img)[:2])
    
    if remove_symbol:
        # removing symbol
        if target_size[0]>720:
            roi_h   = 40 - (crop_dim/target_size[0]*360 if not fill_border else 0)
        else:
            roi_h   = 40 - (crop_dim if not fill_border else 0)
        roi_dim = [int(roi_h/360*target_size[0]), int(40/640*target_size[1])]
        roi = tf.ones((roi_dim[0], roi_dim[1], 3), dtype=tf.float32)
        mask= 1.0 - tf.pad(roi,
                         [[target_size[0]-roi_dim[0],0],
                          [0,target_size[1]-roi_dim[1]],
                          [0,0]])
        new_img = new_img*mask # blackout the symbol
        
    new_img = tf.reshape(new_img, tf.shape(img))
    return new_img


def RandomFlip(img, prob_hflip=0.5, prob_vflip=0.5):
    if random_float()<prob_hflip:
        img = tf.image.flip_left_right(img)
    if random_float()<prob_vflip:
        img = tf.image.flip_up_down(img)
    return img
    
def RandomRotate(img, prob_rotate=0.5):
    if random_float()<prob_rotate:
        img = tf.image.rot90(img, k=random_int([], 0, 4))
    return img

def RandomJitter(img, hue, sat, cont, bri):        
    img = tf.image.random_hue(img, hue)
    img = tf.image.random_saturation(img, sat[0], sat[1])
    img = tf.image.random_contrast(img, cont[0], cont[1])
    img = tf.image.random_brightness(img, bri)
    return img


def mirror_boundary(v, max_v):
    # v % (max_v*2.0-2.0) ==> v % (512*2-2) ==> [0..1022]
    # [0..1022] - (max_v-1.0) ==> [0..1022] - 511 ==> [-511..511]
    # -1.0 * abs([-511..511]) ==> [-511..0]
    # [-511..0] + max_v - 1.0 ==> [-511..0] + 511 ==> [0..511]
    mirror_v = -1.0 * tf.math.abs(
        v % (max_v*2.0-2.0) - (max_v-1.0)) + max_v-1.0
    return mirror_v

def clip_boundary(v, max_v):
    clip_v = tf.clip_by_value(v, 0.0, max_v-1.0)
    return clip_v

def interpolate_bilinear(image, map_x, map_y):
    def _gather(image, map_x, map_y):
        map_stack = tf.stack([map_x, map_y]) # [ 2, height, width ]
        map_indices = tf.transpose(
            map_stack, perm=[1, 2, 0])       # [ height, width, 2 ]
        map_indices = tf.cast(map_indices, dtype=tf.int32)
        gather_image = tf.gather_nd(image, map_indices)
        return gather_image
    
    ll = _gather(image, tf.math.floor(map_x), tf.math.floor(map_y))
    lr = _gather(image, tf.math.ceil(map_x), tf.math.floor(map_y))
    ul = _gather(image, tf.math.floor(map_x), tf.math.ceil(map_y))
    ur = _gather(image, tf.math.ceil(map_x), tf.math.ceil(map_y))
    
    fraction_x = tf.expand_dims(map_x % 1.0, axis=-1) # [h, w, 1]
    int_l = (lr - ll) * fraction_x + ll
    int_u = (ur - ul) * fraction_x + ul
    
    fraction_y = tf.expand_dims(map_y % 1.0, axis=-1) # [h, w, 1]
    interpolate_image = (int_u - int_l) * fraction_y + int_l
    return interpolate_image

def remap(image, height, width, map_x, map_y, mode):
    assert \
        mode in ('mirror', 'constant'), \
        "mode is neither 'mirror' nor 'constant'"

    height_f = tf.cast(height, dtype=tf.float32)
    width_f = tf.cast(width, dtype=tf.float32)
    map_x = tf.reshape(map_x, shape=[height, width])
    map_y = tf.reshape(map_y, shape=[height, width])
    if mode == 'mirror':
        b_map_x = mirror_boundary(map_x, width_f)
        b_map_y = mirror_boundary(map_y, height_f)
    else:
        b_map_x = clip_boundary(map_x, width_f)
        b_map_y = clip_boundary(map_y, height_f)
        
    image_remap = interpolate_bilinear(image, b_map_x, b_map_y)
    
    if mode == 'constant':
        map_stack = tf.stack([map_x, map_y])
        map_indices = tf.transpose(map_stack, perm=[1, 2, 0])
        x_ge_0 = (0.0 <= map_indices[ : , : , 0])    # [h, w]
        x_lt_w = (map_indices[ : , : , 0] < width_f)
        y_ge_0 = (0.0 <= map_indices[ : , : , 1])
        y_lt_h = (map_indices[ : , : , 1] < height_f)
        inside_boundary = tf.math.reduce_all(
            tf.stack([x_ge_0, x_lt_w, y_ge_0, y_lt_h]), axis=0) # [h, w]
        inside_boundary = inside_boundary[ : , : , tf.newaxis]  # [h, w, 1]
        image_remap = tf.where(inside_boundary, image_remap, 0.0)

    return image_remap

def initUndistortRectifyMap(height, width, k, dx, dy):
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    
    f_x = width
    f_y = height
    c_x = width * 0.5 + dx
    c_y = height * 0.5 + dy
    
    f_dash_x = f_x
    c_dash_x = (width - 1.0) * 0.5
    f_dash_y = f_y
    c_dash_y = (height - 1.0) * 0.5

    h_rng = tf.range(height, dtype=tf.float32)
    w_rng = tf.range(width, dtype=tf.float32)
    v, u = tf.meshgrid(h_rng, w_rng)
    
    x = (u - c_dash_x) / f_dash_x
    y = (v - c_dash_y) / f_dash_y
    x_dash = x
    y_dash = y
    
    r_2 = x_dash * x_dash + y_dash * y_dash
    r_4 = r_2 * r_2
    x_dash_dash = x_dash * (1 + k*r_2 + k*r_4)
    y_dash_dash = y_dash * (1 + k*r_2 + k*r_4)

    map_x = x_dash_dash * f_x + c_x
    map_y = y_dash_dash * f_y + c_y
    return map_x, map_y

def OpticalDistortion(image, distort_limit, shift_limit, p=1.0):
    if random_float() > p:
        return image
    k  = random_float([], -distort_limit, distort_limit)
    dx = random_float([], -shift_limit, shift_limit)
    dy = random_float([], -shift_limit, shift_limit)
    image_shape = tf.shape(image)
    height = image_shape[0]
    width  = image_shape[1]
    dim    = [height, width]
    if height!=width:
        mx_dim = tf.math.reduce_max(image_shape)
        image  = tf.image.resize(image, [mx_dim, mx_dim])
        dim    = [mx_dim, mx_dim]
    map_x, map_y = initUndistortRectifyMap(dim[0], dim[1], k, dx, dy)
    aug_image = remap(image, dim[0], dim[1], map_x, map_y, mode='constant')
    if height!=width:
        aug_image = tf.image.resize(aug_image, [height, width])
    aug_image = tf.reshape(aug_image, [height, width, 3])
    return aug_image

def make_grid_distorted_maps(height, width, num_steps, xsteps, ysteps):
    def _make_maps_before_last(size, step, steps): # size=512, step=102,
                                                   # steps.shape=[num_steps]
        step_rep = tf.repeat(step, num_steps)  # [102, 102, 102, 102, 102]
        step_rep_f = tf.cast(step_rep, dtype=tf.float32)
        step_inc = step_rep_f * steps          # [102*s_0, ..., 102*s_4]
        cur = tf.math.cumsum(step_inc)         # [si_0, si_0 + si_1, ... ]
        zero = tf.zeros([1], dtype=tf.float32)
        prev = tf.concat([ zero, cur[ :-1] ], axis=0) # [0, c_0, ..., c_3]
        prev_cur = tf.stack([prev, cur])       # [[p_0, p_1, ...], [c_0, c_1, ...]]
        ranges = tf.transpose(prev_cur)        # [[p_0, c_0], [p_1, c_1], ... ]

        def _linspace_range(rng):
            return tf.linspace(rng[0], rng[1], step)
 
        maps_stack = tf.map_fn(_linspace_range, ranges)
        maps = tf.reshape(maps_stack, [-1])    # [-1] flatten into 1-D
        return maps
    
    def _make_last_map(size, step, last_start):
        last_step = size - step * num_steps  # 512 - 102*5 = 2 
        size_f = tf.cast(size, dtype=tf.float32)
        last_map = tf.linspace(last_start, size_f-1.0, last_step)
        return last_map
    
    def _make_distorted_map(size, steps):
        step = size // num_steps               # step=102 
        maps_before_last = _make_maps_before_last(size, step, steps[ :-1 ])
        last_map = _make_last_map(size, step, maps_before_last[-1])
        distorted_map = tf.concat([maps_before_last, last_map], axis=0)
        return distorted_map

    xx = _make_distorted_map(width, xsteps)
    yy = _make_distorted_map(height, ysteps)
    map_y, map_x = tf.meshgrid(xx, yy)
    return map_x, map_y

def GridDistortion(image, num_steps, distort_limit, p=1.0):
    if random_float() > p:
        return image
    xsteps = tf.random.uniform([num_steps + 1],
                               minval=1.0 - distort_limit,
                               maxval=1.0 + distort_limit)
    ysteps = tf.random.uniform([num_steps + 1],
                               minval=1.0 - distort_limit,
                               maxval=1.0 + distort_limit)

    image_shape = tf.shape(image)
    height = image_shape[0]
    width  = image_shape[1]
    dim    = [height, width]
    if height!=width:
        mx_dim = tf.math.reduce_max(image_shape)
        image  = tf.image.resize(image, [mx_dim, mx_dim])
        dim    = [mx_dim, mx_dim]
    map_x, map_y = make_grid_distorted_maps(dim[0], dim[1], num_steps, xsteps, ysteps)
    aug_image    = remap(image, dim[0], dim[1], map_x, map_y, mode='constant')
    if height!=width:
        aug_image = tf.image.resize(aug_image, [height, width])
    aug_image = tf.reshape(aug_image, [height, width, 3])
    return aug_image