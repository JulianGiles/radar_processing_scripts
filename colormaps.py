""" Contains dictionaries of the colormaps and their values. """

import numpy as np
import os

# Creating a dictionary containing the LangRainbow12 colormap values.
LangRainbow12_data = {
    'blue': [
        (0.0, 0.97000000000000008, 0.97000000000000008),
        (0.090909090909090912, 0.95599999999999996, 0.95599999999999996),
        (0.18181818181818182, 0.94500000000000006, 0.94500000000000006),
        (0.27272727272727271, 0.93700000000000006, 0.93700000000000006),
        (0.36363636363636365, 0.93199999999999994, 0.93199999999999994),
        (0.45454545454545459, 0.92999999999999994, 0.92999999999999994),
        (0.54545454545454541, 0.14900000000000002, 0.14900000000000002),
        (0.63636363636363635, 0.060000000000000053, 0.060000000000000053),
        (0.72727272727272729, 0.042000000000000037, 0.042000000000000037),
        (0.81818181818181823, 0.027000000000000024, 0.027000000000000024),
        (0.90909090909090917, 0.015000000000000013, 0.015000000000000013),
        (1.0, 0.0060000000000000053, 0.0060000000000000053)],
    'green': [
        (0.0, 0.82999999999999996, 0.82999999999999996),
        (0.090909090909090912, 0.7240000000000002, 0.7240000000000002),
        (0.18181818181818182, 0.64799999999999991, 0.64799999999999991),
        (0.27272727272727271, 0.67660000000000009, 0.67660000000000009),
        (0.36363636363636365, 0.76879999999999971, 0.76879999999999971),
        (0.45454545454545459, 0.92999999999999983, 0.92999999999999983),
        (0.54545454545454541, 0.93100000000000005, 0.93100000000000005),
        (0.63636363636363635, 0.75929999999999997, 0.75929999999999997),
        (0.72727272727272729, 0.54600000000000004, 0.54600000000000004),
        (0.81818181818181823, 0.35999999999999999, 0.35999999999999999),
        (0.90909090909090917, 0.20500000000000002, 0.20500000000000002),
        (1.0, 0.08415600000000005, 0.08415600000000005)],
    'red': [
        (0.0, 0.89999999999999991, 0.89999999999999991),
        (0.090909090909090912, 0.77039999999999997, 0.77039999999999997),
        (0.18181818181818182, 0.61499999999999999, 0.61499999999999999),
        (0.27272727272727271, 0.50300000000000011, 0.50300000000000011),
        (0.36363636363636365, 0.38800000000000012, 0.38800000000000012),
        (0.45454545454545459, 0.27000000000000024, 0.27000000000000024),
        (0.54545454545454541, 0.93099999999999983, 0.93099999999999983),
        (0.63636363636363635, 0.89999999999999991, 0.89999999999999991),
        (0.72727272727272729, 0.79800000000000004, 0.79800000000000004),
        (0.81818181818181823, 0.69299999999999995, 0.69299999999999995),
        (0.90909090909090917, 0.58500000000000008, 0.58500000000000008),
        (1.0, 0.4740000000000002, 0.4740000000000002)]
}


# Creating a dictionary of the Homeyer colormap values.
def yuv_rainbow_24(nc):
    path1 = np.linspace(0.8*np.pi, 1.8*np.pi, nc)
    path2 = np.linspace(-0.33*np.pi, 0.33*np.pi, nc)

    y = np.concatenate([np.linspace(0.3, 0.85, nc*2//5),
                        np.linspace(0.9, 0.0, nc - nc*2//5)])
    u = 0.40*np.sin(path1)
    v = 0.55*np.sin(path2) + 0.1

    rgb_from_yuv = np.array([[1, 0, 1.13983],
                             [1, -0.39465, -0.58060],
                             [1, 2.03211, 0]])
    cmap_dict = {'blue': [], 'green': [], 'red': []}
    for i in range(len(y)):
        yuv = np.array([y[i], u[i], v[i]])
        rgb = rgb_from_yuv.dot(yuv)
        red_tuple = (i/(len(y)-1), rgb[0], rgb[0])
        green_tuple = (i/(len(y)-1), rgb[1], rgb[1])
        blue_tuple = (i/(len(y)-1), rgb[2], rgb[2])
        cmap_dict['blue'].append(blue_tuple)
        cmap_dict['red'].append(red_tuple)
        cmap_dict['green'].append(green_tuple)
    return cmap_dict


data_dir = os.path.split(__file__)[0]
bal_rgb_vals = np.genfromtxt(os.path.join(data_dir, 'balance-rgb.txt'))

blue_to_red = {'red':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }


# Making a dictionary of all the colormap dictionaries.
datad = {
        'HomeyerRainbow': yuv_rainbow_24(15),
        'LangRainbow12': LangRainbow12_data,
        'Blue_to_red' : blue_to_red,
        'balance' : bal_rgb_vals}
