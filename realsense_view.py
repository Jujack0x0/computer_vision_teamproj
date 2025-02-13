# import pyrealsense2 as rs
# import numpy as np
# import cv2

# pipeline = rs.pipeline()
# config = rs.config()

# try:
#     pipeline.start(config)
#     print("Pipeline started successfully.")
# except Exception as e:
#     print("Pipeline start error:", e)


# try:
#     # 스트림 설정
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    
#     # 파이프라인 시작
#     pipeline.start(config)
#     print("Pipeline started successfully.")

#     while True:
#         try:
#             frames = pipeline.wait_for_frames(10000)  # 최대 10초 대기
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()
#             if not depth_frame or not color_frame:
#                 continue

#             depth_image = np.asanyarray(depth_frame.get_data())
#             color_image = np.asanyarray(color_frame.get_data())

#             images = np.hstack((color_image, depth_image))
#             cv2.imshow('RealSense Stream', images)

#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#         except RuntimeError as e:
#             print("Frame wait error:", e)

# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()


import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 리얼센스 스트리밍 활성화
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 파이프라인 시작
    pipeline.start(config)

    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 컬러 프레임과 뎁스 프레임 데이터를 numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 뎁스 데이터를 보기 쉽게 조정
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 컬러와 뎁스 영상 결합
            images = np.hstack((color_image, depth_colormap))

            # 영상 표시
            cv2.imshow('RealSense D457 Stream', images)

            # 'q'를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 파이프라인 정지 및 자원 해제
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()