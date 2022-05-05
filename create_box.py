from tools import write_pdb
import random
import numpy as np

def generate_coordinates(elements:dict,dim:int,box) -> None:

	if not isinstance(box,np.ndarray):
		raise TypeError('Box must be a numpy array')

	if dim not in [1,2,3]:
		raise AttributeError('Only available dimensions are 1,2,3')

	N = 0
	for value in elements.values():
		N+=value

	names = []
	positions = np.zeros((N,3))

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

	for item in elements.items():
		for _ in range(item[1]):
			names.append(item[0])
	random.shuffle(names)
	return names, positions

def main():
	elements = {'Ar': 250,'Kr': 250,'Ur': 12}
	dim = 3
	box = np.array([28.98,28.98,28.98])
	#box = np.array([274.95,274.95,274.95])
	names, positions = generate_coordinates(elements,dim,box)

	write_pdb('test_liq.pdb','w',box,names,positions,0)

if __name__ == '__main__':
	main()