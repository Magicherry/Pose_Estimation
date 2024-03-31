import cv2
import math
import numpy as np
import pdb
import os
import copy


# 定义连接线的索引
# mediapipe
# skeleton_lines = [
#     (0, 1), (0, 4), (1, 2), (4, 5), (2, 3), (5, 6), (3, 7), (6, 8), (9, 10),
#     (11, 12), (12, 14), (11, 13), (14, 16), (13, 15), (12, 24), (11, 23),
#     (24, 26), (23, 25), (26, 28), (25, 27),
#     (28, 30), (28, 32), (30, 32), (27, 29), (27, 31), (29, 31)
# ]

# movenet
skeleton_lines = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]
white = (255, 255, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
orange = (0, 165, 255)
red = (0, 0, 255)

def plot_skeleton(image, points):
    points = np.array(points, dtype=np.int32)
    # 绘制骨骼点
    for i in range(len(points)):
        point = points[i][:2]
        if i == 0:
            color = white
        else:
            if i % 2 == 1:
                color = green
            else:
                color = blue
        cv2.circle(image, point, 2, color, 2)
    # 绘制连接线
    for line in skeleton_lines:
        point1 = points[line[0]][:2]
        point2 = points[line[1]][:2]
        cv2.line(image, point1, point2, white, 1)
    return None

# 根据三点坐标计算夹角
def calculate_angle(point_1, point_2, point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1] - point_2[1]))
    tmp = max(-1, min(1, (b*b-a*a-c*c)/(-2*a*c+1e-6)))
    B=math.degrees(math.acos(tmp))
    return B
