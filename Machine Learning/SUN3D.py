import cv2
import numpy as np


def T_transform(depth_1, extrinsic1, extrinsic2, intrinsic, point):
    point_coor = np.ones((3, 1))
    point_coor[:2] = point  # 齐次坐标系
    point_3d = np.linalg.inv(intrinsic).dot(point_coor) * depth_1[point[1], point[0]]
    point3d_coor = np.ones((4, 1))
    point3d_coor[:3, :] = point_3d
    point_3d_ = np.linalg.inv(extrinsic2).dot(extrinsic1.dot(point3d_coor))[:3, :]  # frame1 -> world_frame -> frame2
    point_2d = intrinsic.dot(point_3d_)
    point_2d = point_2d / point_2d[2]
    return point_2d[:2]


def load_extrinsic(filename, frame_id):
    T = np.zeros((4, 4))
    T[3, 3] = 1.

    with open(filename, 'r') as f:
        text_line = f.readlines()
        T1 = text_line[3*(frame_id-1):3*frame_id]
        for ind, line in enumerate(T1):
            text_number = line.split(' ')
            decimal = [float(deci) for deci in text_number if len(deci) > 0]
            row = np.array(decimal)
            T[ind] = row
    return T


def load_intrinsic(filename):
    K = np.zeros((3, 3))

    with open(filename, 'r') as f:
        text_line = f.readlines()
        for ind, line in enumerate(text_line):
            text_number = line.split(' ')
            decimal = [float(deci) for deci in text_number if len(deci) > 0]
            row = np.array(decimal)
            K[ind] = row
    return K


color_1 = cv2.imread('./image/0000001-000000020293.jpg')  # frame_id - time_sequence
color_2 = cv2.imread('./image/0000002-000000053809.jpg')

# trying to find the nearest depth - image pair in time sequence !
depth_1 = cv2.imread('./depth/0000002-000000033369.png', cv2.IMREAD_ANYDEPTH)
depth_2 = cv2.imread('./depth/0000003-000000066739.png', cv2.IMREAD_ANYDEPTH)

depth_1 = (depth_1 >> 3)
depth_2 = (depth_2 >> 3)

select_point = np.array([[89], [212]])  # x, y coordinate in color image

# read intrinsics
K = load_intrinsic("./intrinsics.txt")

T1 = load_extrinsic("./extrinsics/20140112152315.txt", 1)
T2 = load_extrinsic("./extrinsics/20140112152315.txt", 2)

p2d = T_transform(depth_1, T1, T2, K, select_point).astype(np.int32)

cv2.circle(color_1, (select_point[1], select_point[0]), 5, (0, 0, 255), thickness=1)
cv2.circle(color_2, (p2d[1], p2d[0]), 5, (0, 0, 255), thickness=1)

cv2.imshow("color_1", color_1)
cv2.imshow("color_2", color_2)
cv2.waitKey(0)
pass
