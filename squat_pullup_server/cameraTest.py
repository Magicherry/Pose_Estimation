import cv2
import os
from datetime import datetime
import time


def main():
    # 创建一个VideoCapture对象，0表示第一个摄像头（如果有多个摄像头，可以尝试使用1、2等）
    cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 获取视频的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建存储视频的目录
    output_dir = 'result/cameraTest'
    os.makedirs(output_dir, exist_ok=True)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{current_time}.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 循环读取摄像头数据并显示
    frame_num = 0
    while True:
        # 记录开始时间
        t1 = time.time()

        # 读取一帧数据
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            print("无法获取帧")
            break

        frame_num += 1

        # 在帧上添加分辨率的文字
        resolution_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, resolution_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 在帧上添加帧率的文字
        fps_text = f"FPS: {fps}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_num_text = f"Frame_num: {frame_num}"
        cv2.putText(frame, frame_num_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 将帧显示在屏幕上
        cv2.imshow('CameraTest', frame)

        # 写入帧到视频文件
        out.write(frame)

        # 记录结束时间并计算耗时（以毫秒为单位）
        t2 = time.time()
        elapsed_time = round((t2 - t1) * 1000)
        print("frame {} use time {}ms".format(frame_num, elapsed_time))

        # 检查是否按下ESC键，如果是则退出循环
        keyValue = cv2.waitKey(1)  # 捕获键值
        if keyValue & 0xFF == 27:
            break

    # 释放摄像头资源、释放视频写入对象并关闭所有窗口
    print("Video has saved to {}.".format(output_file))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
