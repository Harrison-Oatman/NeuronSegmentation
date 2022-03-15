from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def hextorgb(h):
    """
    converts hexidecimal string to RGB
    """
    h = h.lstrip('#')
    r = int(h[:2],16)
    g = int(h[2:4],16)
    b = int(h[4:],16)
    return [r/255., g/255., b/255.]


def seqfromhexes(points, hs, name='CustomMap'):
    """
    create LinearSegmentedColormap

    ---Arguments---
    points: locations of color changes
    hs: hexidecimal codes before and after the points (length 2*number of poitns)
    """
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, point in enumerate(points):
        for j,col in enumerate(['red','green','blue']):
            c1 = hextorgb(hs[2*i])[j]
            c2 = hextorgb(hs[2*i+1])[j]
            cdict[col].append((point, c1, c2))
    return LinearSegmentedColormap(name, cdict)


def seqfromdict(cdict, name='CustomMap'):
    points = np.sort(list(cdict.keys()))

    hs = []
    for point in points:
        hs.append(cdict[point])
        hs.append(cdict[point])

    # normalize to (0,1)
    if len(points) >= 1:
        points = (points - points[0])/(points[-1] - points[0])

    return seqfromhexes(points, hs, name)


"""
some colors and color maps
"""

ss = '#e6eceb'

default = { "bg" : "#F7F7F7",
        "ax" : "#2E2E2E",
        "c1" : "#41B8A6",
        "c2" : "#B84153",
        "c3" : "#f8a68c"}

dark = { "bg" : "#102e2a",
        "ax" : "#cf7c87",
        "c1" : "#b4cfcb",
        "c2" : "#cf7c87",
        "c3" : "#f8a68c"}
#d3e2e0
light = { "bg" : "#ffd8cc",
        "ax" : "#5c2e35",
        "c1" : "#102e2a",
        "c2" : "#5c2e35",
        "c3" : "#f8a68c"}

slight = { "bg" : "#E8CBB0",
        "ax" : "#9AB4A6",
        "c1" : "#A6E0C1",
        "c2" : "#7EC29E",
        "c3" : "#078541"}

gDefault = seqfromhexes([0.0, 0.0001, 1.0], [default['bg'], default['bg'], default['bg'], default['c2'], default['c3'], default['c3']])
bgDark = seqfromhexes([0.0, 0.0001, 1.0], [dark['bg'], dark['bg'], dark['bg'], dark['c2'], dark['c3'], dark['c3']])
bgLight = seqfromhexes([0.0, 0.0001, 1.0], [light['bg'], light['bg'], light['bg'], light['c2'], light['c3'], light['c3']])
bgsLight = seqfromhexes([0.0, 0.0001, 1.0], [slight['bg'], slight['bg'], slight['bg'], slight['c2'], slight['c3'], slight['c3']])

gradDark = seqfromhexes([0.0, 0.5, 1.0], [dark['bg'], dark['bg'], dark['c2'], dark['c2'], dark['c3'], dark['c3']])
gradLight = seqfromhexes([0.0, 0.5, 1.0], [light['bg'], light['bg'], light['c2'], light['c2'], light['c3'], light['c3']])

skLight = seqfromhexes([0.0, 0.001, 1.0], [light['bg'], light['bg'], light['bg'], light['ax'], light['ax'], light['ax']])