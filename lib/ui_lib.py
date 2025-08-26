# -*- coding: utf-8 -*-
import cv2
import time
import scipy.io as scio
import numpy as np



# ****** image precessing library functions ****** #
def img_calibrate(rsrc, rdst):
    ''' calibrate image to real word view.
    '''
    '''
    rsrc = np.float32([[372, 106],
                        [372, 525],
                        [252, 287],
                        [252, 368]])

    rdst = np.float32([[0, -1.75],
                        [0, 1.75],
                        [60, -1.75],
                        [60, 1.75]])
    '''
    M = cv2.getPerspectiveTransform(np.float32(rdst), np.float32(rsrc))
    return M


def img_rotate(img, angle, theme_flag=0, factor=2):
    ''' Rotate image acoording to the angle.
    '''
    rows, cols, dims = img.shape
    # get ratate matrix
    M = cv2.getRotationMatrix2D((rows/factor, cols/factor), angle, 1)

    # rotate matrix
    bdVal = 255*(1-theme_flag)
    img = cv2.warpAffine(img, M, (rows, cols), borderValue=(bdVal,bdVal,bdVal))
    img = cv2.imencode('.jpg', img)[1]

    return img.tobytes()

def img_rotate_raw(img, angle, theme_flag=0, factor=2):
    ''' Rotate image acoording to the angle.
    '''
    rows, cols, dims = img.shape
    # get ratate matrix
    M = cv2.getRotationMatrix2D((rows/factor, cols/factor), angle, 1)

    # rotate matrix
    bdVal = 255*(1-theme_flag)
    img = cv2.warpAffine(img, M, (rows, cols), borderValue=(bdVal,bdVal,bdVal))

    return img

def img_transform(x, y, M):
    '''
    Instrucs:
    	according the global coordinates, draw the path point in image
    Args:
    	x,   x coordinates
    	y,   y coordinates
        M，  transform matrix
    '''
    srcPts = []

    if x.shape[0]>1:
        for ex, ey in zip(x,y):
            srcPts.append((ex, ey))
    else:
        srcPts=[(x[0],y[0])]
    original = np.array([srcPts], dtype=np.float32)
    dstPts = cv2.perspectiveTransform(original, M)
    return dstPts


def draw_path_new(img, pts, color):
    '''
    Instrucs:
        draw the path in the specific image
    Args:
        img,     specific image as backend
        path_x,  x coordinates of the path
        path_y,  y coordinates of the path
        color,   color of the path points
    '''
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color=color)

    '''
    for row, col in zip(path_x, path_y):
        img = cv2.line(img,(tmp_y, tmp_x),(int(col),int(row)), color, thickness=sz)
        tmp_x = int(row)
        tmp_y = int(col)
        #if row>=0 and row<img.shape[0] and col >=0 and col < img.shape[1]:
        #   img[int(row-sz):int(row+sz), int(col-sz):int(col+sz)] = color
    '''
    return img

def draw_lane_new(scope, coeff, image, M, flag=0, sz=0.05, color=(0,255,0)):
    '''
    Instrucs:
        according the lane coeff ad original image to generate the curve
    Args:
        scope: scope of x coordinate
        coeff: coefficient of the lane path, as cubic equation
        image: backend image that we draw lane curve on
        M    : tranform matrix
        flag : if flag=0, the lane curve is current lane, else, represent aside lane.
        sz   : width of the lane curve
        color: color of the lane curve
    ''' 
    # caculate the [x, y] coordinate
    height, width, dim = image.shape
    #coeff_fix = [-1, -1, -1/2*1.54/100000, -1/6*1/1000000.]

    # limit the band
    if scope[1] > 80:
        scope[1] = 80
    n = 100
    xPos = np.linspace(scope[0], scope[1], n)
    yPos = coeff[0]*(xPos**0) + coeff[1]*(xPos**1) + \
             coeff[2]*(xPos**2) + coeff[3]*(xPos**3) + \
         coeff[4]*(xPos**4) + coeff[5]*(xPos**5) 

    line_x = np.zeros((2*n,))
    line_y = np.zeros((2*n,))

    line_x[0:n]   = xPos
    line_x[n:2*n] = np.flip(xPos,axis=0)

    line_y[0:n]   = yPos - sz
    line_y[n:2*n] = np.flip(yPos + sz,axis=0)


    # deal with perspective transform view

    path_pts = img_transform(line_x, line_y, M)

    pts_draw = path_pts[0].astype(np.int32)


    return draw_path_new(image, np.flip(pts_draw,axis=1), color=color)


    
def draw_path(img, path_x, path_y, color, type='solid', sz=1):
    '''
    Instrucs:
        draw the path in the specific image
    Args:
        img,     specific image as backend
        path_x,  x coordinates of the path
        path_y,  y coordinates of the path
        color,   color of the path points
    '''
    tmp_x = int(path_x[0])
    tmp_y = int(path_y[0])

    frame_cnt = 0
    for row, col in zip(path_x, path_y):
        if color == (0,255,0):
            img = cv2.line(img,(tmp_y, tmp_x),(int(col),int(row)), color, thickness=20)
        elif color == (0,0,255):
            img = cv2.line(img,(tmp_y, tmp_x),(int(col),int(row)), color, thickness=10)
        elif type == 'dotted':
            if frame_cnt % 3 == 0:
                img = cv2.line(img,(tmp_y, tmp_x),(int(col),int(row)), color, thickness=2)
        else:
            img = cv2.line(img,(tmp_y, tmp_x),(int(col),int(row)), color, thickness=2)
        tmp_x = int(row)
        tmp_y = int(col)
        frame_cnt = frame_cnt + 1

    return img

def draw_lane(scope, coeff, image, M, type, flag=0, sz=2, color=(0,255,0)):
	'''
	Instrucs:
		according the lane coeff ad original image to generate the curve
	Args:
		scope: scope of x coordinate
		coeff: coefficient of the lane path, as cubic equation
		image: backend image that we draw lane curve on
        M    : tranform matrix
		flag : if flag=0, the lane curve is current lane, else, represent aside lane.
		sz   : width of the lane curve
		color: color of the lane curve
	'''	
	# caculate the [x, y] coordinate
	height, width, dim = image.shape
	#coeff_fix = [-1, -1, -1/2*1.54/100000, -1/6*1/1000000.]

	# limit the band
	if scope[1] > 80:
		scope[1] = 80
	xPos = np.linspace(scope[0], scope[1], 50)
	yPos = coeff[0]*(xPos**0) + coeff[1]*(xPos**1) + \
		     coeff[2]*(xPos**2) + coeff[3]*(xPos**3) + \
         coeff[4]*(xPos**4) + coeff[5]*(xPos**5) 

	yPos = yPos
	# deal with perspective transform view
	if flag == 0:
		path_pts = img_transform(xPos, yPos, M)
		tmp_x = list(path_pts[0][:,0])
		tmp_y = list(path_pts[0][:,1])

	# deal with bird eye view
	else:
		tmp_x = height - xPos*height/50
		tmp_y = width/2 + yPos*(width/2)/8.
	return draw_path(image, tmp_x, tmp_y, sz = sz, color=color, type=type)
    


def draw_object(img, M, x, y, width=50, height=50, color=(0, 255, 255), sz=2, resize=1,theta =0.2, id_os=0):
    ''' draw radar objects on a image
    '''
    width = width * resize
    height = height * resize


    dx = -x
    dy = -y
    dtheta = -theta
    # T_trans:transform matrix 
    T_trans = np.array([[np.cos(dtheta), np.sin(dtheta), -(dx)], \
                  [-np.sin(dtheta), np.cos(dtheta), -dy], \
                  [0, 0 ,1]
    ])
    tmpX = [-height/2, -height/2, height/2, height/2]
    tmpY = [-width/2,   width/2,  width/2,  -width/2]
    transformdPoints = np.dot(T_trans, np.stack([tmpX, tmpY, [1 for i in range(4)]],axis=0)) 
    #print(x, y)

    tmp_x = transformdPoints[0,:]

    tmp_y = transformdPoints[1,:]

    path_pts = img_transform(tmp_x, tmp_y, M)

    #print(tmp_x, tmp_y, path_pts)
    Px = list(path_pts[0][:,0])
    Py = list(path_pts[0][:,1])

    #print(Px, Py)
    area = np.array([ [Py[0],Px[0]], [Py[1],Px[1]], \
           [Py[2],Px[2]], [Py[3],Px[3]] ], dtype=np.int32)

    #print(area)
    img = cv2.fillPoly(img, [area], color)

    if id_os>0:
        cv2.putText(img, str(id_os), (int(sum(Py)/4.), (int(min(Px) - 10)) ), cv2.FONT_HERSHEY_DUPLEX,  1, (0, 0, 255), 1)

    return img

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, M, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 60.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  path_pts = img_transform(path_x, path_y, M)
  tmp_x = list(path_pts[0][:,0])
  tmp_y = list(path_pts[0][:,1])
  draw_path(img, tmp_x, tmp_y, color)