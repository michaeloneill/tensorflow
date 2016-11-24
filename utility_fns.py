import numpy as np


def unit_scale(X):
        """ 
        Scales all values in X to be between 0 and 1 
        """
        X = X.astype(np.float32)  # copies by default
        X -= X.min()
        X *= 1.0 / X.max()
        return X


def sample_patches(images, patch_dim, num_patches):

    ''' 
    images is [N x H x W x C]
    returns [num_patches x patch_dim x patch_dim x C] 
    '''

    N = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3]  # channels

    patches = np.zeros((num_patches, patch_dim, patch_dim, C))

    # coordinates limit for top left of patch
    max_row_start = H - patch_dim
    max_col_start = W - patch_dim

    for i in xrange(num_patches):
        row_start = np.random.randint(max_row_start + 1)
        col_start = np.random.randint(max_col_start + 1)
        im_idx = np.random.randint(N)
        patches[i, :, :, :] = images[im_idx, row_start:row_start+patch_dim,
                                     col_start:col_start+patch_dim, :]

    return patches

