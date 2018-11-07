import os
import numpy as np
from multiprocessing import Process, Pool
from functools import partial
from nilearn import image
import nibabel as nib

def multiprocessing_resample(img, target_affine):
    resampled_img = image.resample_img(img, target_affine=target_affine,
                                   interpolation='nearest')
    return resampled_img

if __name__ == '__main__':
    subjects = ['age0073', 'age0002', 'age0072', 'age0003', 'age0074',
                'age0004', 'age0075', 'age0007', 'age0084', 'age0008',
                'age0088', 'age0009', 'age0091', 'age0010', 'age0092',
                'age0012', 'age0093', 'age0013', 'age0095', 'age0014',
                'age0096', 'age0017', 'age0097', 'age0018', 'age0098',
                'age0020', 'age0100', 'age0021', 'age0101', 'age0022',
                'age0103', 'age0023', 'age0105', 'age0027', 'age0106',
                'age0028', 'age0110', 'age0029', 'age0111', 'age0030',
                'age0112', 'age0034', 'age0113', 'age0037', 'age0114',
                'age0039', 'age0115', 'age0040', 'age0119', 'age0042']
    # load nibabel images
    home_dir = os.getenv('HOME')
    data_path = os.path.join(home_dir, 'NaN', 'wm_data')
    fileList = os.listdir(data_path)
    imgs = [nib.load(os.path.join(data_path, file)) for file in fileList if
            file[5:12] in subjects]
    target_affine = np.array([[  -3.,    0.,    0.,   90.],
                             [   0.,    3.,    0., -126.],
                             [   0.,    0.,    3.,  -72.],
                             [   0.,    0.,    0.,    1.]])
    # initialise Pool and set target_affine to a fixed value. Imgs is a list to
    # be iterated over
    pool = Pool()
    args = partial(multiprocessing_resample, target_affine=target_affine)
    res = pool.map(args, imgs)

    print(res[0].shape)

