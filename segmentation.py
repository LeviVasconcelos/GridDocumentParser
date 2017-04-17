import cv2
import numpy as np
import random

def withinBounds(p, bounds):
	return p[0] >= 0 and p[0] < bounds[0] and \
		   p[1] >= 0 and p[1] < bounds[1]

def getNeighbours(p, bounds):
	indexes = [ np.array((0,-1)), np.array((1,-1)), 
				np.array((1,0)), np.array((1, 1)), 
				np.array((0,1)), np.array((-1, 1)),
				np.array((-1,0)), np.array((-1,-1))] #Vizinhos de um pixel
	x = [ tuple(p+i) for i in indexes if withinBounds(p+i, bounds)]
	return x

def getMapNeighbours(p, bounds):
	indexes = [ np.array((0, -1)), np.array((-1,-1)),
							np.array((-1,0)), np.array((-1, 1)) ]
	x = [tuple(p+i) for i in indexes if withinBounds(p+i, bounds)]
	return x


def floodFill(img, p, k):
	mask = np.zeros(img.shape, np.int)
	border = []
	stack = [p]
	component = []
	min_cords = np.array([img.shape[0],img.shape[1]])
	max_cords = np.array([0,0])
	while(len(stack)):
		pt = stack.pop()
		for z in getNeighbours(pt, img.shape):
			if (mask[z] == 0):
				mask[z] = 1
				if (img[z] == k):
					stack += [z]
				else:
					border += [z]
		mask[tuple(pt)] = 2;
		min_cords = [min(i) for i in zip(min_cords,pt)]
		max_cords = [max(i) for i in zip(max_cords,pt)]
		component += [pt]
	return [component,border,[min_cords,max_cords]]


class Component:
	def __init__(self):
		self.points = []
		self.label = None
		self.outter = None
		self.rect = []

def createComponent(label, point, outter):
	c = Component()
	c.label = label
	c.points += point
	c.outter = outter
	return c

def mapUnion(img):
	component_dict = {}
	img_map = -1*np.ones(img.shape, np.int)
	for i in range(0,img.shape[0]):
		for j in range(0, img.shape[1]):
			eligibleGroups = []
			outter = None

			for p in getMapNeighbours((i,j), img.shape):
				if (img[p] == img[(i,j)]):
					eligibleGroups += img_map[p]


			if (len(eligibleGroups) == 0):
				img_map[(i,j)] = len(component_dict)
				newComp = createComponent(img_map[(i,j)], (i,j), outter )
				component_dict[img_map[(i,j)]] = newComp

			else:
				fundComponents(component_dict, img_map, (i,j), eligibleGroups)


	return component_dict, img_map



def randColorVect(sz):
	x = [ [random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)] for i in range(0,sz) ]
	return x




IMAGE_TH = 220
img = cv2.imread('BoletoBancario.png', 0)
_, img = cv2.threshold(img, IMAGE_TH, 255, cv2.THRESH_BINARY)
p = [0,0]


img_dict, img_map = mapUnion(img)


'''
c,b,s = floodFill(img, p, img[tuple(p)])
print(s)
print(img.shape)
#print(len(c))
for p in c:
	img[tuple(p)] = 100
cv2.rectangle(img, (s[0][1],s[0][0]), (s[1][1],s[1][0]), 0, 3)
'''
colord_img = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
colors = randColorVect(len(img_dict))
for i in range(0,img.shape[0]):
	for j in range(0,img.shape[1]):
		colord_img[(i,j)] = colors[ img_map[(i,j)] ]

cv2.imshow('teste', colord_img)
cv2.waitKey(0);
cv2.imshow('binarized', img)
cv2.waitKey(0);