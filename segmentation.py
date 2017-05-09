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
				np.array((-1,0)), np.array((-1,-1))] #pixel neighbours
	x = [ tuple(p+i) for i in indexes if withinBounds(p+i, bounds)]
	return x


'''
Just conceptual implementation, not currently being used because of its performance.
MapUnion is faster.
'''
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


#
#-----------------------------------------------------------------------------------------
#Union-set algorithm implementation, modified to fit our needs, but the idea is the same.
#-----------------------------------------------------------------------------------------
#

def getParent(p, img_map):
	while (tuple(img_map[p]) != p):
		p = tuple(img_map[p])
	return p


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


#Computes a pixel's neighbourhood (left and upper pixel)
def getMapNeighbours(p, bounds):
	indexes = [ np.array((0, -1)), np.array((-1,-1)),
							np.array((-1,0)), np.array((-1, 1)) ]
	x = [tuple(p+i) for i in indexes if withinBounds(p+i, bounds)]
	return x



def mapUnion(img):
	component_dict = {}
	img_map = np.zeros((img.shape[0], img.shape[1], 2), np.int)

	for i in range(img.shape[0]): # Visits every pixel
		for j in range(img.shape[1]):
			eligibleGroups = [];

			for p in getMapNeighbours((i,j), img.shape):
				#If pixels in the considered neighbourhood (left and up) of (i,j) has the same color
				#store on eligibleGroups to be grouped later.
				if (img[p] == img[(i,j)] and getParent(p, img_map) not in eligibleGroups):
					eligibleGroups += [getParent(p, img_map)]

			if(len(eligibleGroups) == 0): #if there is no alike group, create a new component for this pixel alone.
				img_map[(i,j)] = (i,j)
				component_dict[(i,j)] = [(i,j)]
			else:
				#Group eligibleGroups adding (i,j) to the new component.
				unionComponents(eligibleGroups, (i,j), img_map, component_dict)
	return component_dict, img_map




def randColorVect(sz):
	x = [ [random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)] for i in range(0,sz) ]
	return x




IMAGE_TH = 220
img = cv2.imread('BoletoBancario.png', 0)

#We should use this algorithm in a binarized image.
_, img = cv2.threshold(img, IMAGE_TH, 255, cv2.THRESH_BINARY)

#initial pixel.
p = [0,0]
cv2.imshow('binarized',img)
cv2.waitKey(0)

print('wait... parsing')
img_dict, img_map = mapUnion(img)
print('Found: ', len(img_dict), ' groups')


print('coloring...')
colord_img = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
colors = randColorVect(len(img_dict))
dict_enum = img_dict.keys()
for i in range(0,img.shape[0]):
	for j in range(0,img.shape[1]):
		p = getParent((i,j), img_map)
		colord_img[(i,j)] = colors[ dict_enum.index(p) ]

cv2.imshow('teste', colord_img)
cv2.waitKey(0);
