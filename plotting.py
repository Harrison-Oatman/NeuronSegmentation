import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib as mpl

"""
Generating Color Maps
"""

def hextorgb(h):
    """
    converts hexidecimal string to RGB
    """
    h = h.lstrip('#')
    r = int(h[:2],16)
    g = int(h[2:4],16)
    b = int(h[4:],16)
    return r/255., g/255., b/255.

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

gDefault = seqfromhexes([0.0, 0.0001,1.0],[default['bg'],default['bg'],default['bg'],default['c2'],default['c3'],default['c3']])
bgDark = seqfromhexes([0.0, 0.0001,1.0],[dark['bg'],dark['bg'],dark['bg'],dark['c2'],dark['c3'],dark['c3']])
bgLight = seqfromhexes([0.0, 0.0001,1.0],[light['bg'],light['bg'],light['bg'],light['c2'],light['c3'],light['c3']])

gradDark = seqfromhexes([0.0, 0.5,1.0],[dark['bg'],dark['bg'],dark['c2'],dark['c2'],dark['c3'],dark['c3']])
gradLight = seqfromhexes([0.0, 0.5,1.0],[light['bg'],light['bg'],light['c2'],light['c2'],light['c3'],light['c3']])

skLight = seqfromhexes([0.0, 0.001,1.0],[light['bg'],light['bg'],light['bg'],light['ax'],light['ax'],light['ax']])

def subplotColored(figsize=(10,10),theme=default,nrows=1,ncols=1):
    # set colors by theme
    mpl.rc('axes',edgecolor=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('xtick',color=theme['ax'])
    mpl.rc('ytick',color=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('figure',facecolor=ss)
    mpl.rc('savefig',facecolor=ss)
    # mpl.rc('figure',dpi=100)
    mpl.rc('savefig',dpi=200)

    return plt.subplots(figsize=figsize,nrows=nrows,ncols=ncols)

"""
KDE plots
"""

def plotKDE(values, title, bw_method = 0.1,
             xlabel = "value", ylabel = "kernel density estimate",
             show = True, savefile = None,figsize=(11,9)):
    return plotKDEs([values], title, None, bw_method, xlabel, ylabel, show, savefile, figsize)

def plotKDEs(valuesList, title, labels = None, bw_methods = 0.1,
             xlabel = "value", ylabel = "kernel density estimate",
             show = True, savefile=None, figsize=(11,9)):

    """
    stack multiple kdeplots
    """

    f, ax = plt.subplots(figsize=figsize)

    legend = True
    if labels is None:
        legend = False
        labels = [None for _ in valuesList]

    if isinstance(bw_methods, float):
        bw_methods = [bw_methods for _ in valuesList]

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    for i, values in enumerate(valuesList):
        sns.kdeplot(x = values, fill=True, bw_method=bw_methods[i], label = labels[i], ax=ax)
    if legend:
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

    if savefile is None:
        savefile = plotpath+title+".png"

    f.savefig(savefile)

    return f, ax

"""
Image Plotting
"""

def implot(image, dir, name = "temp", cmap=bgLight, buffer=None,
           show=True, figsize=(10,10),interpolation='antialiased',
           theme=default, axes=False, title=None,alpha=1.0):
    """
    plotting of an image, configured for my slideshows
    """
    plotImage = image.copy()

    # determine max value
    vmax = np.max(plotImage)

    # set colors by theme
    mpl.rc('axes',edgecolor=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('xtick',color=theme['ax'])
    mpl.rc('ytick',color=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('figure',facecolor=ss)
    mpl.rc('savefig',facecolor=ss)


    fig, ax1 = plt.subplots(figsize=figsize)

    if title is not None:
        fig.suptitle(title, fontsize=25)

    ax1.tick_params(left=axes,
                    bottom=axes,
                    labelbottom=axes,
                    labelleft=axes)

    ax1.imshow(plotImage,cmap=cmap, vmax=vmax, interpolation=interpolation,alpha=alpha)

    plt.savefig(dir + name + ".png")

    if show:
        plt.show()

    return fig, ax1

def embedimg(image, bbs, dir, name = "temp", cmap=bgLight, buffer=None,
           show=True, width=10,interpolation='antialiased',
           theme=default, axticks=False, title=None,alpha=1.0):
    """
    Plots an image with regions inside the image magnified
    """
    imgMax = np.max(image)
    plotImage = image.copy()

    # set title
    if title is None:
        title = name

    # determine max value
    vmax = np.max(plotImage)

    # set colors by theme
    mpl.rc('axes',edgecolor=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('xtick',color=theme['ax'])
    mpl.rc('ytick',color=theme['ax'])
    mpl.rc('text',color=theme['ax'])
    mpl.rc('figure',facecolor=ss)
    mpl.rc('savefig',facecolor=ss)
    # mpl.rc('figure',dpi=200)
    mpl.rc('savefig',dpi=200)

    n = len(bbs)

    fig = plt.figure(figsize=(((n+1)/n)*width, width))

    # fig.suptitle(title, fontsize=25)

    main = plt.subplot2grid((n,n + 1), loc=(0,0), colspan=n, rowspan=n, fig=fig)
    main.imshow(plotImage,cmap=cmap, vmax=vmax, interpolation=interpolation,alpha=alpha)

    # main.axis('off')

    axes = []

    for i, bb in enumerate(bbs):
        # Create a Rectangle patch
        rect = patches.Rectangle((bb[1],bb[0]), bb[3], bb[2], linewidth=1, edgecolor='r', facecolor='none')
        letter = chr(97 + i)
        main.text(bb[1], bb[0], letter, verticalalignment = 'bottom', horizontalalignment='left', color='r',fontsize=20)

        # Add the patch to the Axes
        main.add_patch(rect)

        ax = plt.subplot2grid((n,n + 1), loc=(i,n), fig=fig)
        loc = plotImage[bb[0]:bb[0] + bb[2], bb[1]:bb[1] + bb[3]]
        ax.imshow(loc,cmap=cmap, vmax=vmax, interpolation=interpolation,alpha=alpha)
        ax.text(bb[2]/40,bb[3]/40,letter, verticalalignment = 'top', horizontalalignment='left', color='r',fontsize=20)

        ax.tick_params(left=axticks,
                    bottom=axticks,
                    labelbottom=axticks,
                    labelleft=axticks)

        # ax.axis('off')

        axes.append(ax)

    fig.tight_layout()

    plt.savefig(dir + name + ".png")

    if show:
        plt.show()

    return fig, main, axes

"""
Point plotting
"""

def embedscattercolor(im,xs,ys,c,bbs,s=1,cmap='gist_earth'):
    """
    embedded scatter plot with the option to specify pointwise colors
    """
    fig, main, axes = embedimg(im,bbs,show=False,name='Endpoints',cmap=cmap)

    main.scatter(xs,ys,s=s,c=c)

    for i, ax in enumerate(axes):
        ysi = []
        xsi = []
        csi = []
        for j in range(len(ys)):
            if ys[j] > bbs[i][0] and ys[j] < bbs[i][0] + bbs[i][2]:
                if xs[j] > bbs[i][1] and xs[j] < bbs[i][1] + bbs[i][3]:
                    # print('ere')
                    ysi.append(ys[j])
                    xsi.append(xs[j])
                    csi.append(c[j])
        ysi=np.array(ysi)
        xsi=np.array(xsi)
        csi=np.array(csi)
        ax.scatter(xsi-bbs[i][1], ysi-bbs[i][0],s=s*8,c=csi)

    plt.show()

def embedscattercolornoseries(im,xs,ys,c,bbs,s=1,cmap='gist_earth'):
    """
    embed scatter color with a single color
    """
    fig, main, axes = embedimg(im,bbs,show=False,name='Endpoints',cmap=skLight)

    main.scatter(xs,ys,s=s,c=c)

    for i, ax in enumerate(axes):
        ysi = []
        xsi = []
        for j in range(len(ys)):
            if ys[j] > bbs[i][0] and ys[j] < bbs[i][0] + bbs[i][2]:
                if xs[j] > bbs[i][1] and xs[j] < bbs[i][1] + bbs[i][3]:
                    # print('ere')
                    ysi.append(ys[j])
                    xsi.append(xs[j])
        ysi=np.array(ysi)
        xsi=np.array(xsi)
        ax.scatter(xsi-bbs[i][1], ysi-bbs[i][0],s=s*8,c=c)

    plt.show()

def embedscatter(im,xs,ys,bbs,c=None,s=1,cmap='gist_earth'):
    if c is None:
        c = "k"
    if len(c) == 1:
        embedscattercolornoseries(im, xs, ys, c, bbs, s, cmap)

    else:
        embedscattercolor(im, xs, ys, c, bbs, s, cmap)
