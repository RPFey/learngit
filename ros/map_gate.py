import cv2
import numpy as np


def merge(line1, line2):
    x_cor = np.array([line1[0], line1[2], line2[0], line2[2]])
    y_cor = np.array([line1[1], line2[3], line2[1], line2[3]])
    return np.array([np.min(x_cor), np.min(y_cor), np.max(x_cor), np.max(y_cor)])


def merge_line(lines, gap_theta=0.1, gap_rho=30):
    merge_set = []
    for line in lines:
        if len(merge_set) == 0:
            merge_set.append(line)
        else:
            for index, ele in enumerate(merge_set):
                rho0 = abs(ele[1]*(ele[2]-ele[0])-ele[0]*(ele[3]-ele[1]))/np.sqrt((ele[2]-ele[0])**2+(ele[3]-ele[1])**2)
                theta0 = np.arcsin((ele[3]-ele[1])/np.sqrt((ele[2]-ele[0])**2+(ele[3]-ele[1])**2))
                rho = abs(line[1] * (line[2] - line[0]) - line[0] * (line[3] - line[1])) / np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2)
                theta = np.arcsin((line[3] - line[1]) / np.sqrt((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2))
                if abs(rho-rho0) <= gap_rho and abs(theta-theta0) <= gap_theta:
                    merge_set[index] = merge(line, merge_set[index])
                    break
                else:
                    continue
            else:
                merge_set.append(line)
    merge_set.sort(key=lambda x: (x[2]-x[0])**2+(x[3]-x[1])**2,reverse=True)
    return np.array(merge_set)


def hough(gray):
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(255-gray, 1, np.pi/180, 20, 20, 10)
    line = lines[:, 0, :]  # type: np.array
    line = merge_line(line)
    return line


def harris(gray, thres=100) -> np.array:
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.02)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    y, x = np.where(dst > 0.05*dst.max())
    corner_points = np.array([x, y]).transpose().reshape(-1, 2)

    # merge points
    merge_set = []
    for point in corner_points:
        if len(merge_set) == 0:
            merge_set.append(point)
        else:
            for index, p in enumerate(merge_set):
                dis = (p[0]-point[0])**2 + (p[1]-point[1])**2
                if dis < thres:
                    merge_set[index] = (p+point) / 2
                    break
                else:
                    continue
            else:
                merge_set.append(point)

    return merge_set


if __name__=='__main__':
    src = cv2.imread("initial_map.pgm")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("gate", cv2.WINDOW_AUTOSIZE)
    obstacle_line = hough(gray)
    corner = harris(gray)
    wall = obstacle_line[0]
    cv2.line(src, (wall[0],wall[1]), (wall[2],wall[3]), (0,0,255), 1)
    gate = []
    for x, y in corner:
        x = int(x)
        y = int(y)
        cv2.circle(src, (x, y), 5, (0, 255, 0), 1)
        error = ((wall[0]-wall[2])*y-(wall[1]-wall[3])*(x-wall[0])-wall[1]*(wall[0]-wall[2])) / np.sqrt((wall[0]-wall[2])**2+(wall[1]-wall[3])**2)
        if abs(error) < 10:
            gate.append(np.array([x,y], dtype=np.int16))
    gate.sort(key=lambda x: x[0])
    center = None
    for index in range(len(gate)-1):
        center = (gate[index]+gate[index+1])//2
        roi = 255 - gray[center[1]-5:center[1]+5, center[0]-5:center[0]+5]
        if roi.max() > 10:
            continue
        else:
            break
    else:
        print("can't find gate")
    cv2.circle(src, (int(center[0]), int(center[1])), 5, (255,0,0), 1)
    cv2.imshow("gate", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()