"""
This module holds helper functions for accessing the training and testing data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


CLASSES = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

COLORS = ['C{:d}'.format(i) for i in range(10)]


def load_data(image_fn):
    """Loads the data from disk"""
    img = plt.imread(image_fn)
    xyz = np.fromfile(image_fn.replace(
        '_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])

    proj = load_proj(image_fn.replace('_image.jpg', '_proj.bin'))

    try:
        bbox = load_bbox(image_fn.replace('_image.jpg', '_bbox.bin'))
    except:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)

    return (img, xyz, proj, bbox)


def load_proj(proj_fn):
    proj = np.fromfile(proj_fn, dtype=np.float32)
    proj.resize([3, proj.size // 3])
    return proj


def load_bbox(bbox_fn):
    bbox = np.fromfile(bbox_fn, dtype=np.float32)
    bbox.resize([bbox.size // 11, 11])
    return bbox


def get_all_2D_bbox(bbox, proj, with_extras=False):
    """Will get all 2D bbox for this image"""
    bboxs = []
    for k, b in enumerate(bbox):
        vert_3D, vert_2D, edges, class_id, t, ignore_in_eval = process_bbox(
            b, proj)
        box = get_2D_bbox(vert_2D)
        if with_extras:
            box.extend([class_id, ignore_in_eval])
        bboxs.append(box)
    return bboxs


def project(xyz, proj):
    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]
    return uv


def rot(n, theta):
    """
    Converts axis and angle to a Roation matrix
    See http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    """
    n = n / np.linalg.norm(n, 2)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def get_3D_bbox(p0, p1):
    '''
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    '''
    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e


def get_2D_bbox(vert_2D):
    min_vals = np.min(vert_2D, axis=1).astype(int)
    max_vals = np.max(vert_2D, axis=1).astype(int)
    return [min_vals[0], max_vals[0], min_vals[1], max_vals[1]]


def process_bbox(bbox_, proj):
    """
    Will return the 3D vertices, 2D vertices (image plane), edge connections, class text,
    origin of bbox in 3D coordinates, and ignore_in_eval?
    """
    bbox = np.array(bbox_, copy=True)
    n = bbox[0:3]                 # axis/rotation, norm is the angle
    theta = np.linalg.norm(n)     # angle of rotation
    n /= theta                    # normalizing axis
    R = rot(n, theta)             # generate rotation matrix
    t = bbox[3:6]                 # get homogoneous translation

    sz = bbox[6:9]                                   # length, width, height
    vert_3D, edges = get_3D_bbox(-sz / 2, sz / 2)    # vertices in local frame
    vert_3D = R @ vert_3D + t[:, np.newaxis]         # vertices in world frame
    # 2D vertices image frame
    vert_2D = proj @ np.vstack([vert_3D, np.ones(8)])
    vert_2D = vert_2D / vert_2D[2, :]                # 2D vertices normalized

    class_id = int(bbox[9])
    ignore_in_eval = bool(bbox[10])

    return (vert_3D, vert_2D, edges, class_id, t, ignore_in_eval)


def visualize_image(img, xyz, proj, bbox, bbox_3D=True):
    uv = project(xyz, proj)
    # color mapping, how far away a point is from the car
    clr = np.linalg.norm(xyz, axis=0)
    fig1 = plt.figure(1, figsize=(16, 9))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)
    ax1.axis('scaled')

    for k, b in enumerate(bbox):
        vert_3D, vert_2D, edges, class_id, t, ignore_in_eval = process_bbox(
            b, proj)
        clr = COLORS[k % len(COLORS)]
        if bbox_3D:
            for e in edges.T:
                ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
        else:
            box = get_2D_bbox(vert_2D)
            ax1.add_patch(patches.Rectangle(
                (box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, edgecolor=clr))

    fig1.tight_layout()
    return (fig1, ax1)
