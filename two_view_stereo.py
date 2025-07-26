import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import tqdm


from dataloader import load_middlebury_data
from utils import viz_camera_poses


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max

def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w

    Returns
    -------
    i_R_j, i_T_j: [3,3], [3,1],
        p_i = i_R_j @ p_j + i_T_j
    B: float,
        the baseline
    """

    #====== STUDENT CODE STARTS ======#
    i_R_j = np.matmul(i_R_w, j_R_w.T)
    i_T_j = i_T_w - np.matmul(i_R_w, np.matmul(j_R_w.T, j_T_w))
    C_i_w = -np.matmul(i_R_w.T, i_T_w)
    C_j_w = -np.matmul(j_R_w.T, j_T_w)
    B = np.linalg.norm(C_i_w - C_j_w)

    #====== STUDENT CODE ENDS   ======#

    return i_R_j, i_T_j, B

def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    rgb_i_rect, rgb_j_rect: [H,W,3],[H,W,3],
        the rectified images.
    K_i_corr, K_j_corr: [3,3],[3,3],
        the corrected camera projection matrix. 
    
    NOTE: K_i_corr, K_j_corr have been computed for you! YOU DON'T NEED TO CHANGE THIS.
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    #====== STUDENT CODE STARTS ======#
    
    H_i = np.matmul(np.matmul(K_i_corr,rect_R_i), np.linalg.inv(K_i))
    H_j = np.matmul(np.matmul(K_j_corr,rect_R_j),np.linalg.inv(K_j))
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))
    #====== STUDENT CODE ENDS   ======#
    
    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    p_rect: [3,3],
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    EPS = 1e-8 
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS)

    #====== STUDENT CODE STARTS ======#
    y_axis = (i_T_j / (np.linalg.norm(i_T_j) + EPS)).flatten()  
    x_axis = np.cross(y_axis, np.array([0, 0, 1]))
    x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)
    rect_R_i = np.vstack((x_axis, y_axis, z_axis))

    #====== STUDENT CODE ENDS   ======#

    return rect_R_i

def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    ssd: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    M = src.shape[0]
    N = dst.shape[0]

    diff_squared = (np.expand_dims(src, axis=1) - np.expand_dims(dst, axis=0))**2
    ssd = diff_squared.sum(axis=(2, 3)) 
    #====== STUDENT CODE ENDS   ======#

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    sad: [M,N],
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    M = src.shape[0]
    N = dst.shape[0]

    diff_abs = np.abs((np.expand_dims(src, axis=1) - np.expand_dims(dst, axis=0)))
    sad = diff_abs.sum(axis=(2, 3)) 
    #====== STUDENT CODE ENDS   ======#

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    zncc: [M,N],
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    #====== STUDENT CODE STARTS ======#
    EPS = 1e-8 
    M = src.shape[0]
    N = dst.shape[0]

    print("ORIGINAL SHAPE", (M,N))
       
    mu_src = np.mean(src, axis=1) 
    mu_dst = np.mean(dst, axis=1)  

    src_centered = src - mu_src[:, np.newaxis, :]
    dst_centered = dst - mu_dst[:, np.newaxis, :]

    std_src = np.sqrt(np.mean(src_centered**2, axis=1))  
    std_dst = np.sqrt(np.mean(dst_centered**2, axis=1)) 

    zncc = np.zeros((M, N))

    for ch in range(3): 
        src_ch = src_centered[:, :, ch]  
        dst_ch = dst_centered[:, :, ch]  
        norm_src = src_ch / (std_src[:, ch][:, np.newaxis] + EPS)
        norm_dst = dst_ch / (std_dst[:, ch][:, np.newaxis] + EPS)

        zncc += np.matmul(norm_src, norm_dst.T)

    print("ZNCC SHAPE", zncc.shape)
    #====== STUDENT CODE ENDS   ======#

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    patch_buffer: [H, W, k_size**2, 3],
        The patch buffer for each pixel
    """

    #====== STUDENT CODE STARTS ======#

    H, W, C = image.shape
    print(image.shape)
    
    pad = (k_size - 1) // 2
    
    padded_shape = (H + 2 * pad, W + 2 * pad, C)
    padded_image = np.zeros(padded_shape)
    
    for h in range(H):
        for w in range(W):
            for c in range(C):
                padded_image[h + pad, w + pad, c] = image[h, w, c]
    
    patch_buffer = np.zeros((H, W, k_size * k_size, C))
    
    patch_index = 0
    for dh in range(-pad, pad + 1):
        for dw in range(-pad, pad + 1):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        patch_buffer[h, w, patch_index, c] = padded_image[h + pad + dh, w + pad + dw, c]
            patch_index += 1 
    #====== STUDENT CODE ENDS   ======#
    print(patch_buffer.shape)
    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel,  img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    #====== STUDENT CODE STARTS ======#
    disp_map = np.zeros(rgb_i.shape[:2], dtype=np.float64)
    lr_consistency_mask = np.ones_like(disp_map, dtype=np.float64)
    
    # 1. Compute the patch buffers for both images
    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    # DO NOT DIRECTLY USE: patch_buffer = image2patch(image, k_size), as it may cause errors in the autograder
    
    # 2. Compute the disparity and LR consistency for each pixel

    # 3. Compute disparity candidates

    H, W = rgb_i.shape[:2]
    
    disp_map = np.empty((H, W), dtype=np.float64)
    lr_consistency_mask = np.zeros((H, W), dtype=np.float64)
    
    patches_i = img2patch_func(rgb_i.astype(float) / 255.0, k_size)
    patches_j = img2patch_func(rgb_j.astype(float) / 255.0, k_size)
    
    vi_idx = np.arange(H)
    vj_idx = np.arange(H)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    
    for u in tqdm.tqdm(range(W)):
        buf_i = patches_i[:, u]
        buf_j = patches_j[:, u]
    
        value = kernel_func(buf_i, buf_j)
    
        best_match_right = np.argmin(value, axis=1) 
        disp_map[:, u] = disp_candidates[vi_idx, best_match_right]
    
        best_match_left = np.argmin(value[:, best_match_right], axis=0)
        lr_consistency_mask[:, u] = (best_match_left == vi_idx).astype(np.float64)


    #====== STUDENT CODE ENDS   ======#

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    dep_map: [H,W]
        depth map
    xyz_cam: [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame.
        You may find using np.meshgrid helpful here.
    """

    #====== STUDENT CODE STARTS ======#
    H, W = disp_map.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    dep_map = np.divide(fy * B, disp_map)
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    x = (u - cx) * dep_map / fx
    y = (v - cy) * dep_map / fy
    z = dep_map
    
    xyz_cam = np.dstack((x, y, z))
    #====== STUDENT CODE ENDS   ======#

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    
    NOTE: This can be done with one line of code!
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    #====== STUDENT CODE STARTS ======#

    pcl_world = np.matmul(c_R_w.T, (pcl_cam - c_T_w.T).T).T
    #====== STUDENT CODE ENDS   ======#

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
