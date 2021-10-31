import numpy as np
import tensorflow as tf
import tensorflow.image as transforms

def parse_task(full_task, sep='-'):
    # e.g 'MNIST'   -> {'action': 'none', 'task':'MNIST'}
    # e.g 'p-MNIST' -> {'action': 'p', 'task':'MNIST'}
    if sep not in full_task:
        action, task = 'none', full_task
    else:
        action, task = full_task.split(sep)
    return task, action


def create_data_iter(task, action='none', batch_size=100, return_dict=False,
                     shuff_buffsz=2000, map_npar=tf.data.AUTOTUNE,
                     ds_prefetch=True, ds_cache=True, return_iters=False):
    task = task.upper()

    if task in ['MNIST', 'FMNIST']:
        # Load data
        tf_ds = tf.keras.datasets.mnist if task == 'MNIST' else tf.keras.datasets.fashion_mnist
        x, y, ds = dict(), dict(), dict()
        (x['train'], y['train']), (x['test'], y['test']) = tf_ds.load_data()
        for k in ['train', 'test']:
            x[k] = (x[k]/255.0).astype('float32')

        # Create tensor dataset
        ds = {k: tf.data.Dataset.from_tensor_slices((x[k],y[k])) for k in ['train', 'test']}

        # Perform permutation
        if action.lower() in ['p', 'perm', 'permuted']:
            ds = {k: v.map(permute_gray_image, num_parallel_calls=map_npar) for k,v in ds.items()}

        # Perform image standardization
        ds = {k: v.map(standardize_gray_image, num_parallel_calls=map_npar) for k,v in ds.items()}

        # Prepare for training + batching
        ds['train'] = ds['train'].shuffle(buffer_size=shuff_buffsz)
        ds = {k: v.batch(batch_size) for k,v in ds.items()}

        # Optional to speedup
        if ds_prefetch: ds = {k: v.prefetch(tf.data.AUTOTUNE) for k,v in ds.items()}
        if ds_cache: ds = {k: v.cache() for k,v in ds.items()}

        # # Create iterators
        if return_iters: ds = {k: iter(v) for k,v in ds.items()}

        if return_dict: return ds
        return ds['train'], ds['test']

    else:
        raise('"%s" task not implemented' %(task))

def standardize_gray_image(image, label):
    # see https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean_img, std_img = tf.math.reduce_mean(image), tf.math.reduce_std(image)
    num_els = 1.0 * image.shape[0] * image.shape[1]
    adjusted_stddev = tf.maximum(std_img, 1.0/tf.math.sqrt(num_els))
    image = (image - mean_img)/adjusted_stddev
    return image, label

def permute_gray_image(image, label):
    rimg = tf.reshape(image, [image.shape[0] * image.shape[1]])
    permut = list(np.random.permutation(len(rimg)))
    rimg = tf.gather(rimg, indices=permut)
    image = tf.reshape(rimg, image.shape)
    return image, label

def test_permute(image):
    import matplotlib.pyplot as plt
    # original
    plt.subplot(131)
    plt.imshow(image)
    plt.title('original')
    # permute pixels
    rimg = tf.reshape(image, -1)
    permut = list(np.random.permutation(len(rimg)))
    rimg = tf.gather(rimg, indices=permut)
    plt.subplot(132)
    plt.imshow(tf.reshape(rimg, (28, 28)))
    plt.title('permuted')
    # recover for checking
    rimg = tf.gather(rimg, indices=tf.math.invert_permutation(permut))
    plt.subplot(133)
    plt.imshow(tf.reshape(rimg, (28, 28)))
    plt.title('recovered')
