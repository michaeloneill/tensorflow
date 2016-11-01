import numpy as np
from utility_fns import unit_scale


def images_to_tuple(images):

        """
        Converts images of shape b x H x W x nChannels(<=4) 
        into 4-tuple of shape 4*(b x H*W,), where
        elements represent R, G, B and alpha channels
        and empty trailing channels are padded with None.
        This shape is suitable for PIL plotting.

        """
        nImages = images.shape[0]
        nChannels = images.shape[-1]
        
        return tuple(images[:, :, :, i].reshape((nImages, -1))
                     for i in range(nChannels)) + (None,)*(4-nChannels)



def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_to_unit_interval=False,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted (the plotting assumes values in range [0,1])


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    assert len(tile_shape) == 2
        
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]
    
    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)
                                    
        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # recurrent call
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_to_unit_interval, output_pixel_vals)
        return out_array


    else:
        # we are dealing with one channel

        H, W = img_shape
        Hs, Ws = tile_spacing

        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row*tile_shape[1] + tile_col < X.shape[0]: # still images to plot
                    this_x = X[tile_row*tile_shape[1] + tile_col]
                    if scale_to_unit_interval:
                        this_img = unit_scale(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                        
                    #assert np.amax(this_img)<=1.0 and np.amin(this_img)>=0.0
                    
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row*(H+Hs): tile_row*(H+Hs)+H,
                        tile_col*(W+Ws): tile_col*(W+Ws)+W
                    ] = this_img*c
                    
        return out_array
