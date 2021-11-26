import ctypes
from io import BytesIO

import cv2
import numpy as np
import skimage as sk
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1)


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(x, mode='s&p', amount=c)
    return np.clip(x, 0, 1)


def speckle_noise(x, severity=1):
    c = [0.2, 0.35, 0.45, 0.6, 0.7][severity - 1]
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1)


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(x, sigma=c, multichannel=True)
    return np.clip(x, 0, 1)


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.6, 1, 1), (0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = gaussian(x, sigma=c[0], multichannel=True)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(x.shape[0] - c[1], c[1], -1):
            for w in range(x.shape[1] - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x, sigma=c[0], multichannel=True), 0, 1)


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    kernel = disk(radius=c[0], alias_blur=c[1])
    return np.clip(cv2.filter2D(x, -1, kernel), 0, 1)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x.squeeze(), 0, 1)


def fog(x, severity=1):
    c = [(2.5, 1.7), (2.5, 1.5), (3., 1.4), (3., 1.2), (3.5, 1.1)][severity - 1]
    max_val = x.max()
    n = c[0] * plasma_fractal(wibbledecay=c[1])[:x.shape[0], :x.shape[1]]
    n = n[..., np.newaxis]
    x = x + n
    return np.clip(x * max_val / (max_val + c[0]), 0, 1)


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1)


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    return np.clip(x + c, 0, 1)


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    return np.clip(x * c[0] + c[1], 0, 1)


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = x.squeeze() * 255
    x = x.astype(np.uint8)
    h, w = x.shape[0], x.shape[1]
    ch, cw = int(h * c), int(w * c)
    x = PILImage.fromarray(x.squeeze())
    x = x.resize((ch, cw), PILImage.BOX)
    x = x.resize((h, w), PILImage.BOX)
    return np.array(x)


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = x.squeeze() * 255
    x = x.astype(np.uint8)
    x = PILImage.fromarray(x).convert('RGB')
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    x = np.clip(x, 0, 255) / 255.0
    return x


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    snow_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer, c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='RGB')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer

    x = c[6] * x + (1 - c[6]) * np.maximum(x, x * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1)


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['src/frost/frost1.png',
                'src/frost/frost2.png',
                'src/frost/frost3.png',
                'src/frost/frost4.jpg',
                'src/frost/frost5.jpg',
                'src/frost/frost6.jpg'][idx]
    frost_img = cv2.imread(filename)
    hw = x.shape[0]
    x_start, y_start = np.random.randint(0, frost_img.shape[0] - hw), np.random.randint(0, frost_img.shape[1] - hw)
    frost_img = frost_img[x_start:x_start + hw, y_start:y_start + hw][..., [2, 1, 0]]
    x = np.clip(c[0] * x + c[1] * frost_img / 255, 0, 1)
    return x


def jpeg_compression(x, severity=1):
    c = [18, 15, 12, 10, 7][severity - 1]
    x = x.squeeze() * 255
    x = x.astype(np.uint8)
    x = PILImage.fromarray(x).convert('RGB')
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    x = np.array(x) / 255.0
    # x = sk.color.rgb2gray(x)
    return x


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    # x =  cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m_max = np.max(m, axis=(0, 1))
        m /= np.max(m, axis=(0, 1))
        m[np.logical_or(np.isinf(m), np.isnan(m))] = 0
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR)
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x = x * (1 - m[..., np.newaxis])
        x = np.clip(x + color, 0, 1)
    # x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return x


def elastic_transform(image, severity=1):
    c = [(28 * 2, 28 * 0.7, 28 * 0.1),
         (28 * 2, 28 * 0.08, 28 * 0.2),
         (28 * 0.05, 28 * 0.01, 28 * 0.02),
         (28 * 0.07, 28 * 0.01, 28 * 0.02),
         (28 * 0.12, 28 * 0.01, 28 * 0.02)][severity - 1]
    # image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    image = np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


_corpt_fns = [
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
    snow,
    frost,
    fog,
    brightness,
    contrast,
    elastic_transform,
    pixelate,
    jpeg_compression,
    speckle_noise,
    gaussian_blur,
    spatter,
    saturate
]

corruption_collection = dict()
for fn in _corpt_fns:
    corruption_collection[fn.__name__] = fn
