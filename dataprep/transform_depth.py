import os
import COEFFICIENTS
import cv2
import numpy as np

VALS_SET=False

kinect_path=r'D:\hand_depth_dataset\kinect'
dirs= [ os.path.join(kinect_path,f) for f in os.listdir(kinect_path) if f.startswith('P')]
for dir in dirs:
    print(f'processing {dir} ...')
    annot_dir=os.path.join(dir,'annot')
    img_path=os.path.join(dir,'color')
    depth_path=os.path.join(dir,'depth')

    undistort_annot_dir=os.path.join(dir,'undistort','annot')
    if not os.path.exists(undistort_annot_dir):
        os.makedirs(undistort_annot_dir)
    undistort_img_dir=os.path.join(dir,'undistort','color')
    if not os.path.exists(undistort_img_dir):
        os.makedirs(undistort_img_dir)
    undistort_depth_dir=os.path.join(dir,'undistort','depth')
    if not os.path.exists(undistort_depth_dir):
        os.makedirs(undistort_depth_dir)

    imgs=os.listdir(img_path)
    for img in imgs:
        rgb_img=cv2.imread(os.path.join(img_path,img))
        if not VALS_SET:
            h, w = rgb_img.shape[:2]
            K=COEFFICIENTS.KINECT_K
            distCoeffs = np.concatenate((COEFFICIENTS.KINECT_RADIAL, COEFFICIENTS.KINECT_TANGENTIAL))
            newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))
        
        VALS_SET=True

        undistorted_rgb=cv2.undistort(rgb_img, K, distCoeffs, None, newCameraMatrix)
        cv2.imwrite(os.path.join(undistort_img_dir,img), undistorted_rgb)

        depth_img=cv2.imread(os.path.join(depth_path,img.split('.')[0]+'.png'),cv2.IMREAD_UNCHANGED)
        undistort_depth=cv2.undistort(depth_img, K, distCoeffs, None, newCameraMatrix)
        cv2.imwrite(os.path.join(undistort_depth_dir,img.replace('jpg','png')), undistort_depth)

        seg_path=os.path.join(annot_dir,img.split('.')[0]+'.png')
        if os.path.exists(seg_path):
            seg_mask = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            undistort_seg=cv2.undistort(seg_mask, K, distCoeffs, None, newCameraMatrix)
            cv2.imwrite(os.path.join(undistort_annot_dir,img.replace('jpg','png')), undistort_seg)






