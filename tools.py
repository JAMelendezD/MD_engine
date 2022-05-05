import numpy as np

def write_pdb(name:str,mode:str,box,atoms:list[str],positions,step:int) -> None:
	'''
	Function to write coordinates as a pdb to a file
	'''
	pdb_format = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
	fixed_format = ['ATOM',
					'INDEX',
					'ATYPE',
					'',
					'',
					'A',
					'INDEX',
					'',
					'POSX',
					'POSY',
					'POSZ',
					1.00,
					1.00,
					'AELE',
					'']
	possible_modes = ['w','a']

	if not isinstance(box,np.ndarray):
		raise TypeError('Box must be a numpy array')

	if not isinstance(positions,np.ndarray):
		raise TypeError('atoms must be a numpy array')

	if mode not in possible_modes:
		raise AttributeError('mode not available only modes are w and a')

	with open(name,mode) as f:
		f.write('CRYST1 {:8.3f}{:8.3f}{:8.3f}{:8.2f}{:8.2f}{:8.2f}\n'.format(*box,90,90,90))
		f.write('MODEL {:>8d}\n'.format(step))
		for i in range(len(atoms)):
			fixed_format[1] = i+1
			fixed_format[2] = atoms[i]
			fixed_format[4] = atoms[i]
			fixed_format[6] = i+1
			fixed_format[8] = positions[i][0]
			fixed_format[9] = positions[i][1]
			fixed_format[10] = positions[i][2]
			fixed_format[13] = atoms[i]
			f.write(pdb_format.format(*fixed_format))
		f.write('TER\n')
		f.write('ENDMDL\n')
	return

def nonblank(f):
	'''
	Generator of non-blank lines
	'''
	for l in f:
		line = l.rstrip()
		if line:
			yield line

def read_itp(name:str) -> dict:
	'''
	Function to read the force field
	'''
	dic = {}
	with open(name,'r') as f:
		for line in nonblank(f):
			data = line.split()
			if data[0] == 'atom':
				dic[data[1]] = (float(data[2]),float(data[3]),float(data[4]),float(data[5]))
	return dic


def read_mdp(name:str) -> dict:
	'''
	Function to initialize the simulation options and to modify them based in the mdp file
	'''
	dic = {'nsteps'		:	0,
			'dt'		:	1,
			'T'			:	298.0,
			'itp'		: 'noble.itp',
			'vdw-cut'	:	9.0,
			'ensemble'  :	1,
			'save'		:	100,}

	options = list(dic.keys())

	with open(name,'r') as f:
		for line in nonblank(f):
			data = line.split()
			if data[0] in options:
				dic[data[0]] = type(dic[data[0]])(data[1].strip())
			else:
				print(f'Warning: option {data[0]} provided in the mdp is not implemented will be ignored')
	return dic

def splitm(line):
	'''
	Function to correctly split a line of a pdb file
	'''
	return([line[0:6].strip(),line[6:11].strip(),line[12:16].strip(),line[16:17].strip(),line[17:20].strip(),
			line[21:22].strip(),line[22:26].strip(),line[26:27].strip(),line[30:38].strip(),line[38:46].strip(),
			line[46:54].strip(),line[54:60].strip(),line[60:66].strip(),line[76:78].strip(),line[78:80].strip()])