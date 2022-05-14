from tools import write_pdb
import random
import numpy as np

def generate_coordinates(elements,box,mode):

	N = 0
	for value in elements.values():
		N+=value

	positions = np.zeros((N,3))

	if box[1] == 0.0:
		dim = 1
	elif box[2] == 0.0:
		dim = 2
	else:
		dim = 3

	for i in range(N):
		if dim == 1:
			f = N
			positions[i][0] = (i%f+0.5)*box[0]/f
			positions[i][1] = 0.0
			positions[i][2] = 0.0
		if dim == 2:
			f = int(N**(1/2))
			positions[i][0] = (i%f+0.5)*box[0]/f
			positions[i][1] = ((i//f)%f+0.5)*box[1]/f
			positions[i][2] = 0.0
		if dim == 3:
			f = int(N**(1/3))+1
			f2 = f*f
			positions[i][0] = (i%f+0.5)*box[0]/f
			positions[i][1] = ((i//f)%f+0.5)*box[1]/f
			positions[i][2] = (int(i/f2)+0.5)*box[2]/f

	if mode == 0:
		names = []
		for item in elements.items():
			for _ in range(item[1]):
				names.append(item[0])
		random.shuffle(names)
	elif mode == 1:
		lists = []
		for item in elements.items():
			lists.append(item[1]*[item[0]])
		names = [item for sublist in zip(*lists) for item in sublist]
	return names, positions

def main():
	mode = 0
	elements = {'Ar':256,'He':256}
	box = np.array([28.98,28.98,28.98])
	#box = np.array([40.98,40.98,40.98])
	#box = np.array([274.95,274.95,274.95])
	names, positions = generate_coordinates(elements,box,mode)
	f = open('test.pdb','w')
	write_pdb(f,box,names,positions,0)
	f.close()
if __name__ == '__main__':
	main()