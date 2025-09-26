import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import cv2

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi, preserve_range=True)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape

    assert c1 == c2
    
    if h1 < h2:
        crop_amount = h2 - h1
        start_crop = int(np.floor(crop_amount/2.))
        end_crop = int(np.ceil(crop_amount/2.))
        if end_crop == 0:
            im2 = im2[start_crop:, :, :]
        else:
            im2 = im2[start_crop:-end_crop, :, :]
    elif h1 > h2:
        crop_amount = h1 - h2
        start_crop = int(np.floor(crop_amount/2.))
        end_crop = int(np.ceil(crop_amount/2.))
        if end_crop == 0:
            im1 = im1[start_crop:, :, :]
        else:
            im1 = im1[start_crop:-end_crop, :, :]
    
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    
    if w1 < w2:
        crop_amount = w2 - w1
        start_crop = int(np.floor(crop_amount/2.))
        end_crop = int(np.ceil(crop_amount/2.))
        if end_crop == 0:
            im2 = im2[:, start_crop:, :]
        else:
            im2 = im2[:, start_crop:-end_crop, :]
    elif w1 > w2:
        crop_amount = w1 - w2
        start_crop = int(np.floor(crop_amount/2.))
        end_crop = int(np.ceil(crop_amount/2.))
        if end_crop == 0:
            im1 = im1[:, start_crop:, :]
        else:
            im1 = im1[:, start_crop:-end_crop, :]
    
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    derek_image = plt.imread('project2_images/DerekPicture.jpg')
    nutmeg_image = plt.imread('project2_images/nutmeg.jpg')
    pts = get_points(nutmeg_image, derek_image)
    print("Derek and Nutmeg pts:")
    print(pts)
    
    broccoli_image = plt.imread('project2_images/broccoli.jpeg')
    tree_image = plt.imread('project2_images/tree.jpeg')
    pts = get_points(broccoli_image, tree_image)
    print("Broccoli and Tree pts:")
    print(pts)

    airplane_image = plt.imread('project2_images/airplane.jpeg')
    eagle_image = plt.imread('project2_images/bird.jpg')
    pts = get_points(airplane_image, eagle_image)
    print("Airplane and Eagle pts:")
    print(pts)

    popsicle_image = cv2.imread('project2_images/chocolate_popsicle.jpg', cv2.IMREAD_COLOR)
    popsicle_image = cv2.cvtColor(popsicle_image, cv2.COLOR_BGR2RGB)
    popsicle_image = popsicle_image.astype(np.float64) / 255.0
    popsicle_image = sktr.rotate(popsicle_image, -10, preserve_range=True)
    popsicle_image = popsicle_image[200:-200, 200:-200, :]

    domo_image = cv2.imread('project2_images/domo.jpeg', cv2.IMREAD_COLOR)
    domo_image = cv2.cvtColor(domo_image, cv2.COLOR_BGR2RGB)
    domo_image = domo_image.astype(np.float64) / 255.0
    domo_image = np.pad(domo_image, ((100, 100), (0, 0), (0, 0)), mode='constant')
    domo_image = cv2.resize(domo_image, (popsicle_image.shape[1], popsicle_image.shape[0]))

    pts = get_points(popsicle_image, domo_image)
    print("Popsicle and Domo pts:")
    print(pts)