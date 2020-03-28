import cv2 as cv
import numpy as np 
#import matplotlib.pyplot as plt 
from math import sqrt
#import pptk
fps = 20


# def paintorbs(img1, kp1, img2, kp2, matches):
#   img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#   plt.imshow(img3)
#   plt.show()

def divide(img, nums):
  shape = np.array(img).shape
  sections = []
  # DO NOT CHANGE ORDER wOR THAT H/W IS EVENLY SPLIT
  # CORRECT ORB PT DENORM DEPENDS ON LOOP ORDER
  for y in range(nums):
    for x in range(nums):
      offsety = (shape[0] // nums) * y
      offsetx = (shape[1] // nums) * x
      leny = shape[0] // nums
      lenx = shape[1] // nums

      imageslice = img[offsety:offsety + leny, offsetx:offsetx + lenx]
      sections.append(imageslice)
    #  sl = img[]

  return np.array(sections)

def doorb(img1, img2):
  orb = cv.ORB_create()
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
  (kp1, desc1) = orb.detectAndCompute(img1, None)
  (kp2, desc2) = orb.detectAndCompute(img2, None)
  if(desc1 is not None and desc2 is not None):
    matches = bf.match(desc1, desc2)
    return kp1, kp2, matches
  return None, None, None

# def show3dhist(pts):
#   dpts = [ [pt[0], pt[1], 1.0] for pt in pts ]
#   v = pptk.viewer(dpts)
#   v.set(point_size=0.01)

class Video:

  def __init__(self, fpath):
    self.cap = cv.VideoCapture(fpath)
    self.front = self.cap.read()[1]
    self.back = None
    self.fronttraces = None
    self.ret = True

  def move(self): # get fresh frame into 
    self.back = self.front
    self.ret, self.front = self.cap.read()
    
   # self.front = self.cap.read()[1] # skip 1 frame

  def show(self):
    #cv.imshow('Frame', self.front)
    #cv.waitKey(0) & 0xFF == ord('q')
    #cv.waitKey(1)
    return

  def gaussian(self):
    self.front = cv.GaussianBlur(self.front, (3,3), 0.5)

  def grey(self):
    self.front = cv.cvtColor(self.front, cv.COLOR_BGR2GRAY)

  def sharpen(self):
    self.front = cv.filter2D(
      self.front,
      -1,
      np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]], np.float32) 
    )
 

  def normalize(self):
    self.front = cv.normalize(
      self.front, 
      None,
      alpha=0,
      beta=255,
      norm_type=cv.NORM_MINMAX
    )

  def _genTrace(self, pt1s, pt2s):
    deltas = np.array(pt1s) - np.array(pt2s)
    deltas = [ (d[0], d[1]) for d in deltas ] # to tuple
    self.fronttraces = list(zip(pt1s, pt2s, deltas))
   # deltas = [ (d[0], d[1], 1.0) for d in deltas ]
    #v = pptk.viewer(deltas)
    #v.set(point_size=0.01)


  def canny(self):
    can = cv.Canny(self.front, 100,200)
    can = np.reshape(can, (can.shape[0], can.shape[1], 1))
    self.front = can 
   # self.front = cv.equalizeHist(self.front)
  def paintorb(self):
    frontsecs = divide(self.front, 3)
    backsecs = divide(self.back, 3)
    secs = list(zip(frontsecs, backsecs))

    kp1persec = []
    kp2persec = []
    matchespersec = []

    numsecs = len(secs)
    secLenY = len(frontsecs[0])
    secLenX = len(frontsecs[0][0])
  
    for idx, sec in enumerate(secs):
      (kp1,kp2,ms) = doorb(sec[0], sec[1])
      if(kp1 is not None and kp2 is not None and ms is not None):
        # since we fed ORB sections, they kps are all in their respective coordinate system
        # => bring keypoint into whole image's coord system

        # 0 to sqrt(numsecs) - 1
        # eg. 0 to 2 each if 9 sections
        sectionIdX = idx % sqrt(numsecs)
        sectionIdY = idx // sqrt(numsecs)
        
        global_kp1 = []
        global_kp2 = []
        for kp in kp1:
          # offset coords back wrt to whole image, not a section of it
          newkp = kp
          newkp.pt = (
            kp.pt[0] + (sectionIdX * secLenX), 
            kp.pt[1] + (sectionIdY * secLenY)
            )
          global_kp1.append(newkp)

        for kp in kp2:
          newkp = kp
          # scale pt coords back onto whole image
          newkp.pt = (
            kp.pt[0] + (sectionIdX * secLenX), 
            kp.pt[1] + (sectionIdY * secLenY)
            )
          global_kp2.append(newkp)

      #  if(idx == 2):
        kp1persec.append(global_kp1)
        kp2persec.append(global_kp2)
        matchespersec.append(ms)
      #  print(len(ms), " matches in sector seen")
  

    kp1kp2matchespersec = list(zip(kp1persec, kp2persec, matchespersec))
  
    pt1s = []
    pt2s = []
    for kp1kp2m in kp1kp2matchespersec:
      kp1 = kp1kp2m[0]
      kp2 = kp1kp2m[1]
      ms = kp1kp2m[2]
    # print(np.array(matches).shape)
      for m in ms:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
      # print(p1, p2)
        if(m.distance < 30):
          cv.line(
            self.front, 
            (int(p1[0]), int(p1[1])), # to int
            (int(p2[0]), int(p2[1])), # to int
            (0,0,255), 
            1
          )
          pt1s.append(p1)
          pt2s.append(p2)

    self._genTrace(pt1s, pt2s)
   # show3dhist(histdata)
    # fig,ax = plt.subplots()
    # distances = [ m.distance for m in matches ]
    # n,bins,patches = ax.hist(
    #   distances,
    #   10, # num bins 
    #   density=max(distances)
    # )
    # plt.show()
    
    # show orbs
    #paintorbs(self.front, kp1, self.back, kp2, matches)




v = Video('./speedchallenge/data/train.mp4')
v.move()
tracearr = []
numframes = 20400
doneframes = 0

while(v.ret):
 # v.canny()
 # v.grey()
  #v.gaussian()
  v.sharpen()
  v.gaussian()
  v.normalize()
  v.paintorb()
  tracearr.append(v.fronttraces)
  v.show()
  v.move()
  doneframes += 1
  print(doneframes / numframes * 100.0 ," percent done")

tracearr = np.array(tracearr)
np.save('save_train', tracearr)
print("Saved")
