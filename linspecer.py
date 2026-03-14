import numpy as np
from scipy.interpolate import PchipInterpolator


def linspecer(N, style=None):

    if N <= 0:
        return np.array([])

    qual_flag = False
    colorblind_flag = False

    if style is not None:
        style = style.lower()

        if style in ['qualitative', 'qua']:
            qual_flag = True

        elif style in ['sequential', 'seq']:
            return colorm(N)

        elif style in ['white', 'whitefade']:
            return white_fade(N)

        elif style in ['red', 'blue', 'green', 'gray', 'grey']:
            return white_fade(N, style)

        elif style == 'colorblind':
            colorblind_flag = True

    set3 = color_brew_to_mat(np.array([
        [141,211,199],
        [255,237,111],
        [190,186,218],
        [251,128,114],
        [128,177,211],
        [253,180,98],
        [179,222,105],
        [188,128,189],
        [217,217,217],
        [204,235,197],
        [252,205,229],
        [255,255,179]
    ]))

    set1 = brighten(color_brew_to_mat(np.array([
        [55,126,184],
        [228,26,28],
        [77,175,74],
        [255,127,0],
        [152,78,163]
    ])),0.8)

    if N == 1:
        return np.array([[55,126,184]])/255

    if N <= 5:
        return set1[:N]

    return colorm(N)


def color_brew_to_mat(arr):
    return arr / 255.0


def brighten(arr, frac=0.9):
    return arr * frac + (1-frac)


def dim(arr, f):
    return arr * f


def cmap2linspecer(arr):
    return arr


def colorm(n=100):

    frac = 0.95

    cmapp = np.array([
        [158,1,66],
        [213,62,79],
        [244,109,67],
        [253,174,97],
        [254,224,139],
        [255*frac,255*frac,191*frac],
        [230,245,152],
        [171,221,164],
        [102,194,165],
        [50,136,189],
        [94,79,162]
    ])

    x = np.linspace(1,n,cmapp.shape[0])
    xi = np.arange(1,n+1)

    cmap = np.zeros((n,3))

    for i in range(3):
        interp = PchipInterpolator(x, cmapp[:,i])
        cmap[:,i] = interp(xi)

    cmap = np.flipud(cmap/255)

    return cmap


def white_fade(n=100, color='blue'):

    if color in ['gray','grey']:
        cmapp = np.array([
            [255,255,255],
            [240,240,240],
            [217,217,217],
            [189,189,189],
            [150,150,150],
            [115,115,115],
            [82,82,82],
            [37,37,37],
            [0,0,0]
        ])

    elif color == 'green':
        cmapp = np.array([
            [247,252,245],
            [229,245,224],
            [199,233,192],
            [161,217,155],
            [116,196,118],
            [65,171,93],
            [35,139,69],
            [0,109,44],
            [0,68,27]
        ])

    elif color == 'blue':
        cmapp = np.array([
            [247,251,255],
            [222,235,247],
            [198,219,239],
            [158,202,225],
            [107,174,214],
            [66,146,198],
            [33,113,181],
            [8,81,156],
            [8,48,107]
        ])

    elif color == 'red':
        cmapp = np.array([
            [255,245,240],
            [254,224,210],
            [252,187,161],
            [252,146,114],
            [251,106,74],
            [239,59,44],
            [203,24,29],
            [165,15,21],
            [103,0,13]
        ])

    else:
        raise ValueError("Color not recognized")

    return interpomap(n, cmapp)


def interpomap(n, cmapp):

    x = np.linspace(1,n,cmapp.shape[0])
    xi = np.arange(1,n+1)

    cmap = np.zeros((n,3))

    for i in range(3):
        interp = PchipInterpolator(x, cmapp[:,i])
        cmap[:,i] = interp(xi)

    return cmap/255
