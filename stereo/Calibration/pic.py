import cv2 as cv

capL = cv.VideoCapture(0)
capR = cv.VideoCapture(2)

count = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR): break
    cv.imshow('Left', frameL[:, ::-1])
    cv.imshow('Right', frameR[:, ::-1])
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        print(f"Saving: {count=}")
        cv.imwrite(f'left/{count}.jpg', frameL)
        cv.imwrite(f'right/{count}.jpg', frameR)
        count += 1

capL.release()
capR.release()
cv.destroyAllWindows()
    