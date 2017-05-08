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
	c.points += [point]
	c.outter = outter
	return c


def changeMapComponents(img_map, comp, newComp):
	for i in comp.points:
		img_map[i] = newComp

def fundComponents(comp_dict, img_map, p, eGroups):
	aux = [ len(comp_dict[i].points) for i in eGroups ]
	biggerGroupIdx = aux.index(max(aux))
	biggerGroup = eGroups[biggerGroupIdx]

	img_map[p] = biggerGroup
	comp_dict[biggerGroup].points += [p]
	comp_dict[biggerGroup].points += sum([comp_dict[i].points for i in eGroups if i != biggerGroup],[])

	for i in eGroups:
		if i != biggerGroup:
			changeMapComponents(img_map, comp_dict[i], biggerGroup)
			del comp_dict[i]


def union(c1, c2, img_map, component_dict):
	root1 = getParent(c1, img_map)
	root2 = getParent(c2, img_map)
	if (root1 == root2):
		return
	img_map[c2] = root1
	component_dict[root1] += component_dict[root2]
	del component_dict[root2]



def unionComponents(groups, p, img_map, component_dict):
	g = groups[0]
	groups = groups[1:] if len(groups) > 0 else []
	img_map[p] = getParent(g, img_map)
	component_dict[getParent(g,img_map)] += [p]
	for g in groups:
		if g != p:
			union(p, g, img_map, component_dict)



def getParent(p, img_map):
	while (tuple(img_map[p]) != p):
		p = tuple(img_map[p])
	return p


def mapUnion_uSet(img):
	component_dict = {}
	img_map = np.zeros([] + img.shape + 2, np.int)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			eligibleGroups = [];

			for p in getMapNeighbours((i,j), img.shape):
				if (img[p] == img[(i,j)] and getParent(p, img_map) bot in eligibleGroups):
					eligibleGroups += [getParent(p, img_map)]

			if(len(eligibleGroups) == 0): #if there is no alike group
				img_map[(i,j)] = (i,j)
				component_dict[(i,j)] = [(i,j)]
			else:
				unionComponents(eligibleGroups, (i,j), img_map, component_dict)




'''
*******************
*******************
MAKE IT RUN FASTER (UNION SET LIKE)
'''
def mapUnion(img):
	component_dict = {}
	img_map = -1*np.ones([] + img.shape + 3, np.int)
	for i in range(0, img_map.shape[0]):
		for j in range(0, img_map.shape[1]):
			img_map = [i, j, 0];
	new_group_idx = 0
	for i in range(0,img.shape[0]):
		for j in range(0, img.shape[1]):
			eligibleGroups = []
			outter = None

			for p in getMapNeighbours((i,j), img.shape):
				if (img[p] == img[(i,j)] and img_map[p][2] not in eligibleGroups):
					eligibleGroups += [img_map[p][2]]

			if (len(eligibleGroups) == 0):
				img_map[(i,j)][2] = new_group_idx
				img_map[(i,j)][0] = i
				img_map[(i,j)][1] = j
				new_group_idx += 1
				newComp = createComponent(img_map[(i,j)], (i,j), outter)
				component_dict[img_map[(i,j)][2]] = newComp
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
cv2.imshow('binarized',img)
cv2.waitKey(0)
print('wait... parsing')
img_dict, img_map = mapUnion(img)

#img_test = np.zeros((10,10), np.uint8)
#img_test[0:5,5] = 255
#img_test[0:180,70] = 255
#img_test[0:90,40:42] = 255
#img_test[0:30,20:22] = 255
#img_test[45,0:100] = 255

#img_dict, img_map = mapUnion(img_test)
#print('max:',np.max(img_map))
#print('min:',np.min(img_map))
#print(img_map[40:45,40:45])
#print(img_test[40:45,40:45])
#print(len(img_dict))
#print(img_map[0:5,0:5])
'''
colord_img = np.zeros([img_test.shape[0], img_test.shape[1], 3], np.uint8)
colors = randColorVect(np.max(img_map)+2)
for i in range(0,img_test.shape[0]):
	for j in range(0,img_test.shape[1]):
		colord_img[(i,j)] = colors[ img_map[(i,j)] ]


cv2.imshow('img_test', colord_img)
cv2.waitKey(0)
'''
print('coloring...')
colord_img = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
colors = randColorVect(np.max(img_map)+2)
for i in range(0,img.shape[0]):
	for j in range(0,img.shape[1]):
		colord_img[(i,j)] = colors[ img_map[(i,j)] ]

cv2.imshow('teste', colord_img)
cv2.waitKey(0);
cv2.imshow('binarized', img)
cv2.waitKey(0);