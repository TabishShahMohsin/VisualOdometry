# from hough_corner_detector import HoughCornerDetecter
# import cv2
# import numpy as np
# from pnp_grid_ransac import ransac

# HEIGHT = 15 # in centimeters
# WIDTH = 10
# fx = fy = 800
# cx = cy = 400
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# distCoeffs = np.zeros((4, 1), dtype=np.float32) # Don't change this line, this was after hours of debugging

# def run_hough_lines_on_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     # cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Couldn't open video.")
#         return

#     processor = HoughCornerDetecter()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video

#         intersections = processor.process_image(frame)
#         for x, y in intersections:
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#         print("intersection_count: ", len(intersections))
#         rvec, tvec = ransac(intersections)

#         t = np.rint(tvec % np.array([[10], [15], [1]])).astype(np.int16)
#         r = np.rint(np.rad2deg(rvec) % np.array([180])).astype(np.int16)
#         print('t', t,'\n', 'r',  r)
        
#         for i in points:
#             cv2.circle(frame, (int(i[0]), int(i[1])), 7, (255, 0, 0), -1)
#         cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        

#         cv2.imshow('Hough Corner Detection', frame)
#         cv2.waitKey(0)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_hough_lines_on_video('test1.mp4')  # Change to your video file path