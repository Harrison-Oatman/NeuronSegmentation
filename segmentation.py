import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image, skeletonize

"""
Finding endpoints, branching points, and crossing points
"""
def kernel_sum(img, xs, ys):
    n, m = img.shape
    outputimg = np.zeros((n,m))

    # determine necessary padding
    lpad = max(0,-min(xs))
    rpad = max(xs)
    tpad = max(0,-min(ys))
    bpad = max(ys)

    padimg = img.copy()

    # add padding
    if lpad > 0:
        padimg = np.hstack((np.zeros((m,lpad)), padimg))
    if rpad > 0:
        padimg = np.hstack((padimg, np.zeros((m,rpad))))
    if tpad > 0:
        padimg = np.vstack((np.zeros((tpad,padimg.shape[1])), padimg))
    if bpad > 0:
        padimg = np.vstack((padimg, np.zeros((bpad,padimg.shape[1]))))


    # take sum
    for x, y in zip(xs, ys):
        outputimg += padimg[:,lpad+x:lpad+m+x][tpad+y:tpad+n+y,:]

    return outputimg

def sq_kernel(img, r):

    ys = []
    xs = []

    for y in range(-r, r+1):
        for x in range(-r, r+1):
            ys.append(y)
            xs.append(x)

    return kernel_sum(img, xs, ys)

def find_endpts(img):
    """
    massively speeds up finding endpoints, splits, crosses
    """

    workingimg = np.array(img) != 0

    outimg = sq_kernel(workingimg, 1)

    endpts = outimg == 2
    splits = outimg == 4
    crosses = outimg == 5

    antisplit = np.zeros(splits.shape)

    for x in [-1,1]:
        xs = [x,x]
        ys = [-1,0]
        antisplit += kernel_sum(workingimg, xs, ys) > 1
        ys = [0,1]
        antisplit += kernel_sum(workingimg, xs, ys) > 1

    for y in [-1,1]:
        ys = [y,y]
        xs = [-1,0]
        antisplit += kernel_sum(workingimg, xs, ys) > 1
        xs = [0,1]
        antisplit += kernel_sum(workingimg, xs, ys) > 1

    splits = np.bitwise_and(antisplit==0,splits)

    endpts = np.bitwise_and(endpts,workingimg)
    splits = np.bitwise_and(splits,workingimg)
    crosses = np.bitwise_and(crosses,workingimg)

    return endpts, splits, crosses

"""
Skeleton management
"""

def rm_nubs(skel, min_length = 10):
    """
    removes endpoint branches of skeleton below a minimum length
    """
    m = min_length

    # pad zeros to newskel for function use
    newskel = np.pad(skel, 1)
    newskel[newskel > 0] = 1

    # find location of endpts, splits, crosses
    endpts, splits, crosses = find_endpts(skel)

    # get x and y values of endpts
    endptys, endptxs = np.where(endpts==1)
    eplocs = zip(endptys, endptxs)

    print(endptys)
    print(endptxs)

    shorties = []

    # iterate through each endpt
    for epy, epx in eplocs:
        branch = [(epy,epx)]
        short = False

        # go for a number of steps equal to minimum length
        for i in range(m):
            cy, cx = branch[-1]

            for y in range(cy-1,cy+2):
                for x in range(cx-1,cx+2):
                    if newskel[y + 1, x + 1] == 1:
                        if (y,x) not in branch:
                            # print(y,x)
                            if splits[y,x] == 1 or crosses[y,x] == 1:
                                short = True
                                break
                            else:
                                branch.append((y,x))

        if short:
            shorties.append(branch)

    print(f'{len(shorties)} short branches found')

    return shorties

def nby2_convert(pts):
    return np.vstack(np.where(pts)).T

def merge_pts(pts, r):
    # print(pts)
    n = pts.shape[0]

    # get pairwise distance
    pwdist = squareform(pdist(pts))
    iis, jjs =  np.where(pwdist < r)

    rmids = set()
    merge_pts = None

    for i, j in zip(iis,jjs):
        if i < j:
            newpt = np.floor(np.mean(pts[[i,j],:], axis=0)).astype(pts.dtype)
            rmids.add(i)
            rmids.add(j)
            if merge_pts is not None:
                merge_pts = np.vstack((merge_pts, newpt.reshape(1,2)))
            else:
                merge_pts = newpt.reshape(1,2)

    rmids = np.array(list(rmids))
    if len(rmids) > 0:
        pts = np.delete(pts, rmids,axis=0)

    return pts, merge_pts

def cross_close_branches(bps, cps, r):
    a = nby2_convert(bps)
    c = nby2_convert(cps)

    newbps = np.zeros(bps.shape)
    newcps = np.zeros(cps.shape)

    while True:
        a, b = merge_pts(a, r)
        if b is None:
            break
        else:
            c = np.vstack((c,b))
    for x, y in a:
        newbps[x,y] = 1
    for x, y in c:
        newcps[x,y] = 1
    return newbps, newcps

def get_crossing_mask(img, cps):
    iimg = img.copy()
    iimg[iimg>0] = 1
    iimg = 1 - iimg

    mask = np.zeros(img.shape)
    n, m = img.shape
    for cpy, cpx in cps:
        if iimg[cpy, cpx] > 0:
            continue

        for r in range(3,20,1):
            # get bounds for square around crossing point
            ymin = max(0,cpy-r)
            xmin = max(0,cpx-r)
            ymax = min(n,cpy+r)
            xmax = min(m,cpx+r)
            space = iimg[ymin:ymax,:][:,xmin:xmax]

            # determine crossing point in space
            y = cpy - ymin
            x = cpx - xmin

            space = space + 0
            ls = label(space)
            props = regionprops(ls)
            # since this is a crossing point, we should have four background regions
            if 4 <= len(props) <= 5:

                areas = [props[i].area > 5 for i in range(len(props))]
                if sum(areas) < 4:
                    continue

                # calculate distance of each point to crossing point
                yy, xx = np.meshgrid(np.arange(space.shape[1]), np.arange(space.shape[0]))
                darr = np.sqrt((yy - y)**2 + (xx - x)**2)

                # print(darr)
                # print(space)
                # store the closest point in each background region
                nearpts = None

                # go through each background region
                for lab in range(1, np.max(ls) + 1):

                    # create a mask so the only contenders are in the region
                    pmask = darr.copy()
                    pmask[ls!=lab] = np.inf

                    # find closest point
                    closey, closex = np.unravel_index(np.argmin(pmask),pmask.shape)
                    new = np.array([[closey,closex]])
                    if nearpts is None:
                        nearpts = new
                    else:
                        nearpts = np.vstack((nearpts, new))

                cpmask = np.zeros(space.shape)
                cpmask[nearpts[:,0],nearpts[:,1]] = 1
                cpmask = convex_hull_image(cpmask)
                cpmask[cpmask>0] = 1
                # implot(cpmask)

                mask[ymin:ymax,:][:,xmin:xmax] += cpmask

                break
    # implot(mask)
    # implot(img-mask > 0,interpolation='nearest')
    # print(np.unique(img-mask))

    return mask

def get_branch_mask(img, bps):
    iimg = img.copy()
    iimg[iimg>0] = 1
    iimg = 1 - iimg

    mask = np.zeros(img.shape)
    n, m = img.shape
    for bpy, bpx in bps:
        for r in range(2,80,1):
            # get bounds for square around crossing point
            ymin = max(0,bpy-r)
            xmin = max(0,bpx-r)
            ymax = min(n,bpy+r)
            xmax = min(m,bpx+r)
            space = iimg[ymin:ymax,:][:,xmin:xmax]

            # determine crossing point in space
            y = bpy - ymin
            x = bpx - xmin

            # label space, locating each region of the background
            ls = label(space)

            # since this is a branch point, we should have three background regions
            if len(regionprops(ls)) == 3:

                # calculate distance of each point to branch point
                yy, xx = np.meshgrid(np.arange(space.shape[1]), np.arange(space.shape[0]))
                darr = np.sqrt((yy - y)**2 + (xx - x)**2)

                # go through each background region
                for lab in range(1,4):

                    # mask for the specific point
                    bpmask = np.zeros(space.shape)

                    # cover up noncontenders in the region
                    pmask = darr.copy()
                    pmask[ls!=lab] = np.inf

                    # find closest point
                    closey, closex = np.unravel_index(np.argmin(pmask),pmask.shape)

                    # mask the line from branch point to nearest region point
                    bpmask[y,x] = 1
                    bpmask[closey,closex] = 1

                    bpmask = convex_hull_image(bpmask)
                    mask[ymin:ymax,:][:,xmin:xmax] += bpmask

                break
    mask[mask > 0] = 1
    # implot(mask)
    # implot(img-mask > 0,interpolation='nearest')
    # print(np.unique(img-mask))

    return mask

def trim_skel(skel, min_length=5):
    shorties = rm_nubs(skel, min_length)
    for short in shorties:
        for y, x in short:
            skel[y,x] = 0
    return skel

def break_down(image):
    """
    break an image down into segments, applying crossing and branch masks
    """
    sampleskel = skeletonize(image,method="lee")
    shorts = rm_nubs(sampleskel,10)

    for short in shorts:
        for y, x in short:
            sampleskel[y,x] = 0

    sep, sbp, scp = find_endpts(sampleskel)


    a,b = cross_close_branches(sbp, scp, 12)


    samplemask = get_crossing_mask(image, nby2_convert(b))
    samplemask += get_branch_mask(image, nby2_convert(a))
    samplemask[samplemask > 0] = 1

    segmentedsampleimage = image - samplemask
    segmentedsampleimage[segmentedsampleimage<1] = 0
    segmentedsampleimage[segmentedsampleimage>=1] = 1

    return segmentedsampleimage, sampleskel
