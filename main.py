import copy
import math
import os
import queue
import signal
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree

from movenet.movenet import Movenet
from utils.utils import plot_skeleton


def frame_reader(video_path, input_queue):
    # 从本地读取视频
    cap = cv2.VideoCapture(video_path)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # 获取原视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取原视频窗口大小
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"分辨率: {width} x {height} 帧率: {fps}")
    frame_num = 0
    while cap.isOpened() and frame_reader_flag:
        # 读视频帧
        ret, frame = cap.read()
        if ret:  # 判断是否读取成功
            input_queue.put([frame_num, frame])
            frame_num += 1
        else:
            input_queue.put([-1, None])
            break
    if video_path == 0:
        cap.release()


def cosin_distance_matching(pose_vector1, pose_vector2):  # 余弦相似度计算函数
    pose_end = 2 * keypoint_num
    pose_vector1 = pose_vector1[0:pose_end]
    pose_vector2 = pose_vector2[0:pose_end]
    # t0 = time.time()
    # cosine_sim = np.dot(pose_vector1, pose_vector2) / (norm(pose_vector1, 2)*norm(pose_vector2, 2))
    cosine_sim = cosine_similarity(np.reshape(pose_vector1, (1, -1)), np.reshape(pose_vector2, (1, -1)))
    distance = 2 * (1 - cosine_sim[0][0])
    return math.sqrt(distance)


def weighted_distance_matching(pose_vector1, pose_vector2):  # 权重相似度计算函数
    pose_end = 2 * keypoint_num
    score_end = pose_end + keypoint_num
    vector1_pose_xy = pose_vector1[0:pose_end]
    vector1_confidences = pose_vector1[pose_end:score_end]
    vector1_confidence_sum = pose_vector1[-1]
    vector2_pose_xy = pose_vector2[0:pose_end]
    summation2 = 0
    for i in range(len(vector1_pose_xy)):
        conf_ind = math.floor(i / 2)
        temp_sum = vector1_confidences[conf_ind] * abs(vector1_pose_xy[i] - vector2_pose_xy[i])
        summation2 = summation2 + temp_sum
    summation1 = 1 / vector1_confidence_sum
    return summation1 * summation2


def get_pose_vector_from_pose_detector(results, height, width):  # 骨骼点归一化函数
    pose = np.array([])
    pose_vector_final = None
    pose = results  # 17个点 * [x, y, conf]
    minx, miny = map(int, np.min(pose[:, :2], axis=0))  # 求出所有点的最小x值和最小y值
    pose_vector = copy.deepcopy(pose)
    pose_vector[:, :2] = pose_vector[:, :2] - np.array([minx, miny])  # 减去最小的x和y，相当于把可能出现在画面中的任意人体骨骼点整体挪到画面左上角
    pose_vector = np.reshape(pose_vector[:, :2], (-1))  # len=51
    pose_vector_norm = np.linalg.norm(pose_vector, 2)
    pose_vector_l2 = pose_vector / pose_vector_norm  # 归一化，相当于把画面中可能不同大小的人体都缩放到同一尺度上，解决近大远小的问题
    score_vector = pose[:, 2]
    score_vector_sum = np.reshape(np.sum(score_vector), -1)  # reshape成一维向量
    pose_vector_final = np.concatenate(
        (pose_vector_l2, score_vector, np.array(score_vector_sum)))  # 52 = 17*2 + 17*1 + 1
    return pose, pose_vector_final


def get_pose_vector_from_one_file(detector, image_path):  # 将base_data中的图变成骨骼点向量
    image = cv2.imread(image_path)
    results = detector.predict(image, verbose=False)
    pose, pose_vector = get_pose_vector_from_pose_detector(results, image.shape[0], image.shape[1])
    return pose, pose_vector


def make_search_tree(detector, base_data_path, action_image_num, metric):  # 创建搜索树，并在此指定用哪个相似度计算函数
    base_pose_list = []
    base_pose_category_list = []
    base_data_vectors = []
    for i in ["down", "up"]:
        for j in range(action_image_num):
            image_path = os.path.join(base_data_path, i, "{}.jpg".format(j))
            print(image_path)
            if not os.path.exists(image_path):
                print("error: " + image_path + " not found.")
            else:
                base_pose_category_list.append(i)
                pose, pose_vector = get_pose_vector_from_one_file(detector, image_path)
                base_pose_list.append(pose)
                base_data_vectors.append(pose_vector)
    base_data_vectors = np.asarray(base_data_vectors)
    tree = BallTree(base_data_vectors, leaf_size=2, metric=metric)
    return tree, base_pose_list, base_pose_category_list


def find_best_match(pose_vector, tree):  # 寻找最佳匹配
    if pose_vector is None:
        return {"index": -1, "dist": -1}
    else:
        pose_vector = np.reshape(pose_vector, (1, -1))
        dist, index = tree.query(pose_vector)  # 匹配到的base_data的相似度距离和序号
        return {"index": index[0][0], "dist": dist[0][0]}


def filter_match(match, dist_thresh, action_image_num):  # 按照相似度阈值进行过滤，大于阈值的认为不够像，就过滤掉
    filtered_match = copy.deepcopy(match)
    if filtered_match["dist"] > dist_thresh:
        filtered_match["index"] = -1
        filtered_match["dist"] = -1
    return filtered_match


def get_body_center(pose):  # 获取身体骨骼中心
    # if len(pose) == 0:
    #     return None
    body_center = np.array([0., 0.])
    body_list = [11, 12, 23, 24, 25, 26, 27, 28]
    for i in body_list:
        body_center += pose[i, :2]
    body_center = np.array(body_center / len(body_list), dtype=np.int32)
    return body_center


def increament_squat_count(pose_counter, pose_category):  # 按照当前输入姿态的类别，将其压入队列
    if len(pose_counter) == 0:
        pose_counter.append([pose_category, 1])
    elif pose_counter[-1][0] == pose_category:
        pose_counter[-1][1] += 1  # [[up,3]]
    else:
        pose_counter.append([pose_category, 1])  # [[up,3], [down,15], [up,15], [down,1], [up,10], [down,10]]
    return pose_counter


def count_total_reps(count, pose_counter, pose_num_1rep_thresh, category_num, verbose=False):  # 过滤掉可能抖动的情况，进行次数统计
    # step 0
    if verbose:
        print(0, pose_counter)
    # step 1
    pose_counter_tmp = [[pose_category, pose_num_1rep] for pose_category, pose_num_1rep in pose_counter \
                        if pose_num_1rep > pose_num_1rep_thresh]  # [[down,15], [up,15], [up,10], [down,10], [up,20]]
    if verbose:
        print(1, pose_counter_tmp)
    # step 2
    pose_counter_refined = []
    for pose_category, pose_num_1rep in pose_counter:
        if len(pose_counter_refined) == 0:
            pose_counter_refined.append([pose_category, pose_num_1rep])
        elif pose_counter_refined[-1][0] == pose_category:
            pose_counter_refined[-1][1] += pose_num_1rep
        else:
            pose_counter_refined.append(
                [pose_category, pose_num_1rep])  # [[up,50], [down,15], [up,25], [down,10], [up,20]]
    count_tmp = max(0, math.floor((len(pose_counter_refined) - 1) / category_num))  # 2
    if verbose:
        print(2, pose_counter_refined)
    # update_score
    if count < count_tmp:
        count = count_tmp
        update_score_flag = True
    else:
        update_score_flag = False
    return count, update_score_flag  # 返回计算后的个数，以及是否完成一次的标志位


def update_best_squat_score(pose, best_squat_score):
    # dist_y_ratio
    mean_hip = (pose[11, :2] + pose[12, :2]) / 2  # 左右髋的中点，下面同理
    mean_knee = (pose[13, :2] + pose[14, :2]) / 2
    mean_ankle = (pose[15, :2] + pose[16, :2]) / 2
    ankle_hip_dist_y = abs(mean_ankle[1] - mean_hip[1])  # 踝到髋的垂直距离
    ankle_knee_dist_y = abs(mean_ankle[1] - mean_knee[1])  # 踝到膝的垂直距离
    dist_y_ratio = min(ankle_knee_dist_y / (ankle_hip_dist_y + 1e-6), 1)  # 加上一个很小的数1e-6，防止分母为0
    # dist_x_ratio  没用到
    shoulder_dist_x = abs(pose[5][0] - pose[6][0])
    knee_dist_x = abs(pose[15][0] - pose[16][0])
    dist_x_ratio = shoulder_dist_x / knee_dist_x if knee_dist_x > shoulder_dist_x else knee_dist_x / shoulder_dist_x
    # squat_score
    squat_score = int(dist_y_ratio * 100)
    best_squat_score = max(squat_score, best_squat_score)
    return best_squat_score


def update_best_pull_up_score(pose, best_pull_up_score):
    mean_elbow_y = (pose[7][1] + pose[8][1]) / 2
    mean_wrist_y = (pose[9][1] + pose[10][1]) / 2
    face_center = np.array([0, 0], dtype=np.float32)
    for i in range(5):
        face_center += pose[i][:2]
    face_center /= 5  # 取脸上5个点的中点
    face_center_y = face_center[1]
    if face_center_y > mean_elbow_y:  # 当脸点低于肘点时是0分，没拉上去
        pull_up_score = 0
    elif face_center_y < mean_wrist_y:  # 当脸点高于腕点是100分
        pull_up_score = 100
    else:
        pull_up_score = int((mean_elbow_y - face_center_y) / (
                mean_elbow_y - mean_wrist_y + 1e-6) * 100)  # 在中间区域，也就是脸点正好在小臂的范围内，脸越靠近腕点分数越高
    best_pull_up_score = max(pull_up_score, best_pull_up_score)
    return best_pull_up_score


# 主函数
def main():
    # 初始化参数
    use_camera = True
    if not use_camera:
        sport_type = "pull_up"  # squat pull_up
        video_path = "video/{}/0.mp4".format(sport_type)
        result_path = "result/{}".format(sport_type)
        video_result_path = "{}/0_res.mp4".format(result_path)
    else:
        sport_type = "squat"  # squat pull_up 起始为深蹲
        video_path = 0

        # 获取当前日期时间
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 生成新的文件名
        result_dir = "result/camera/"
        video_result_path = os.path.join(result_dir, f"camera_res_{current_time}.mp4")

    match_dist_thresh = 0.3  # 相似度阈值，适合余弦相似度函数
    global frame_reader_flag
    frame_reader_flag = True  # 是否取帧的标志位
    # input_video_h, input_video_w = None, None  # 输入帧的尺寸以及写视频的尺寸
    draw_res = True  # 是否画上检测结果
    draw_pose = True  # 是否画上检测到的骨骼点

    # 模型初始化
    model_name = "movenet_lightning"  # movenet_lightning  movenet_thunder
    pose_detector = Movenet(model_name)

    # 建立搜索树
    global keypoint_num
    keypoint_num = 17  # movenet可以检测人体17个关键点，具体见17pose.png
    action_image_num = 2  # 每个运动类别中每个姿态图的比照数量
    base_data_path = "base_data/squat"
    tree_1, base_pose_list_1, base_pose_category_list_1 = make_search_tree(pose_detector, base_data_path,
                                                                           action_image_num,
                                                                           metric=cosin_distance_matching)
    base_data_path = "base_data/pull_up"
    tree_2, base_pose_list_2, base_pose_category_list_2 = make_search_tree(pose_detector, base_data_path,
                                                                           action_image_num,
                                                                           metric=cosin_distance_matching)
    if sport_type == "squat":
        tree = tree_1
        base_pose_category_list = base_pose_category_list_1
    elif sport_type == "pull_up":
        tree = tree_2
        base_pose_category_list = base_pose_category_list_2
    print("make search tree done.")

    # 另开一个线程用于加载视频帧
    input_queue = queue.Queue()
    t = threading.Thread(target=frame_reader, args=(video_path, input_queue))
    t.start()

    # 评分初始化
    best_squat_score = 0
    best_pull_up_score = 0
    best_score = 0

    # 拉流
    cap = cv2.VideoCapture(video_path)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # 获取原视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取原视频窗口大小
    input_video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_result_path, fourcc, fps,
                                   (input_video_w, input_video_h))  # 写视频的参数设置

    # 主循环
    count = 0
    change_sport_type = False  # 是否切换运动类别
    no_player_cnt = 0
    no_player_cnt_max = 60
    pose_counter = []  # 用于统计运动次数的队列
    frame_num = -1
    while True:
        t1 = time.time()
        # 读视频帧
        ret, frame = cap.read()
        if ret:  # 判断是否读取成功
            frame_num += 1

            # 当使用摄像头且按键切换运动类别时，这些参数和变量要重置
            if use_camera and change_sport_type:
                change_sport_type = False
                count = 0
                update_score_flag = False
                best_score = 0
                pose_category = None
                no_player_cnt = 0
                pose_counter = []
                if sport_type == "squat":
                    tree = tree_1
                    base_pose_category_list = base_pose_category_list_1
                elif sport_type == "pull_up":
                    tree = tree_2
                    base_pose_category_list = base_pose_category_list_2

            results = pose_detector.predict(frame, verbose=False)  # 人体骨骼点推理
            pose, pose_vector = get_pose_vector_from_pose_detector(results, frame.shape[0], frame.shape[1])  # 归一化操作
            pose_category = None
            if len(pose) > 0:
                match = find_best_match(pose_vector, tree)  # 最相似的匹配
                filtered_match = filter_match(match, match_dist_thresh, action_image_num)  # 匹配结果过滤
                if filtered_match["index"] != -1:
                    pose_category = base_pose_category_list[filtered_match["index"]]  # 判断当前姿态类别
                    pose_counter = increament_squat_count(pose_counter, pose_category)  # 姿态的状态统计
                    count, update_score_flag = count_total_reps(count, pose_counter, 10, 2, False)  # 计数更新
                    if update_score_flag:  # 当完成一次时，更新最佳成绩，用于显示
                        if sport_type == "squat":
                            best_score = best_squat_score
                            print("squat_count: {}".format(count))
                            print("best_squat_score: {}".format(best_squat_score))
                            best_squat_score = 0
                        elif sport_type == "pull_up":
                            best_score = best_pull_up_score
                            print("pull_up_count: {}".format(count))
                            print("best_pull_up_score: {}".format(best_pull_up_score))
                            best_pull_up_score = 0
                    else:  # 当还没完成一次时，计算当前评分并给出截止到目前的本次最佳评分
                        if sport_type == "squat":
                            best_squat_score = update_best_squat_score(pose, best_squat_score)
                        elif sport_type == "pull_up":
                            best_pull_up_score = update_best_pull_up_score(pose, best_pull_up_score)
                no_player_cnt = 0
            else:  # 未用上，因为movenet总会在画面中得到一组骨骼点，哪怕没有人，这个是单人movenet的缺陷
                no_player_cnt += 1
                if no_player_cnt >= no_player_cnt_max:  # 当画面中没人超过60帧时，代码停止运行
                    video_writer.write(frame)
                    frame_reader_flag = False
                    break

            t2 = time.time()
            # print("frame {} use time {}ms".format(frame_num, round((t2-t1)*1000))) # 打印每帧处理速度
            process_fps = int(1 / (t2 - t1))
            if draw_res:
                line_interval = 40
                line = 70
                cv2.putText(frame, "Sport Type: {}".format(sport_type), (30, line), 0, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                line += line_interval
                cv2.putText(frame, "Frame Num: {}".format(frame_num), (30, line), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
                line += line_interval
                cv2.putText(frame, "Process Fps: {}".format(process_fps), (30, line), 0, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                line += line_interval
                cv2.putText(frame, "Category: {}".format(pose_category), (30, line), 0, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                line += line_interval
                cv2.putText(frame, "Count: {}".format(count), (30, line), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
                line += line_interval
                cv2.putText(frame, "Score: {}".format(best_score), (30, line), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 画骨骼点
            if draw_pose:
                plot_skeleton(frame, pose)
            # 写视频帧
            video_writer.write(frame)
            # 显示视频
            cv2.imshow("frame", frame)
            cv2.setWindowTitle("frame", "Pose Estimation")
            # 按键退出
            keyValue = cv2.waitKey(1)  # 捕获键值
            if keyValue & 0xFF == 27:  # esc键退出程序
                frame_reader_flag = False  # 停止读取视频
                break
            elif keyValue & 0xFF == ord('1') and use_camera:  # 1键 切换为 深蹲
                if sport_type != "squat":
                    sport_type = "squat"
                    change_sport_type = True
            elif keyValue & 0xFF == ord('2') and use_camera:  # 2键 切换为 引体向上
                if sport_type != "pull_up":
                    sport_type = "pull_up"
                    change_sport_type = True

    # 释放资源
    print("{} process done, save to {}.".format(video_path, video_result_path))
    video_writer.release()
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
