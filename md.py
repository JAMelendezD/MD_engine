import numpy as np
from numba import jit,prange
import tools as tl
from scipy.interpolate import interp1d as interp
from scipy.special import erf
import os
import time
import argparse

def generate_velocities(T,mass):
	kB = 1.38e-23
	amu = 1.660539*1e-27
	mass = mass*amu
	a = np.sqrt(kB*T/mass)
	v = np.arange(0,25000,0.1)
	cdf = erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a
	inv_cdf = interp(cdf,v) 
	rand_num = np.random.random(1)
	speed = inv_cdf(rand_num)*1e-5 # convert from m/s to A/fs
    
	theta = np.arccos(np.random.uniform(-1,1,1))
	phi = np.random.uniform(0,2*np.pi,1)
    
	vx = speed * np.sin(theta) * np.cos(phi)
	vy = speed * np.sin(theta) * np.sin(phi)
	vz = speed * np.cos(theta)
	return [vx[0], vy[0], vz[0]]

def read_box(pdb,T,ff):
	positions = []
	velocities = []
	atoms = []
	masses = []
	known = list(ff.keys())
	with open(pdb,'r') as f:
		N = 0
		for line in f:
			data = tl.splitm(line)
			if data[0] == 'ATOM':
				if data[2] not in known:
					raise KeyError('Atom in structure file not found in the itp file')
				positions.append(data[8:11])
				atoms.append(data[2])
				mass = ff[data[2]][2]
				masses.append(mass)
				velocities.append(generate_velocities(T,mass))
			if data[0] == 'CRYST1':
				data = line.split()
				box = np.array(data[1:4],dtype=float)
	N = len(positions)
	return N, atoms, np.array(positions,dtype=float), np.array(velocities,dtype=float), box, np.array(masses,dtype=float)

@jit(nopython=True)
def lj(r,rvec,C6,C12):
	sr6 = 1.0/r**6
	sr8 = 1.0/r**8
	sr14 = 1.0/r**14
	pot = C12*sr6*sr6-C6*sr6
	force = (12*C12*sr14-6*C6*sr8)*rvec
	return pot,force

@jit(nopython=True,fastmath=True,parallel = True)
def compute_forces(poss,box,C6s,C12s,masses,cutoff):
	N = len(poss)
	amu = 1.660539*1e-27
	energies = np.zeros(N)
	forces = np.zeros((N,3))
	accs = np.zeros((N,3))
	for i in prange(N):
		energy = 0 
		force = np.zeros(3)
		vectors = np.remainder(poss[i] - poss + box[0]/2.0, box[0]) - box[0]/2.0
		for j in range(N):
			if i != j:
				rvec = vectors[j]
				r = np.sqrt(np.dot(rvec,rvec))
				#if r <= cutoff:
				e,f = lj(r,rvec,C6s[i][j],C12s[i][j])
				energy += e
				force += f
		energies[i] = energy
		forces[i] = force
		accs[i] = force/(masses[i]*1000)
	return energies, forces, accs

@jit(nopython=True)
def update_velocities(N,vels,old_accs,accs,dt2):
	for i in range(N):
		for j in range(3):
			vels[i][j] = vels[i][j]+(old_accs[i][j]+accs[i][j])*dt2
	return vels

@jit(nopython=True)
def update_positions(N,box,poss,vels,accs,dt,dt3):
	for i in range(N):
		for j in range(3):
			new_pos = poss[i][j]+vels[i][j]*dt+accs[i][j]*dt3
			if new_pos > box[j]:
				poss[i][j] = new_pos-box[j]
			elif new_pos < 0:
				poss[i][j] = new_pos+box[j]
			else:
				poss[i][j] = new_pos  
	return poss

def generate_lj_params(atoms,ff):
	N = len(atoms)
	C6s = np.zeros((N,N))
	C12s = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			sig_ij = (ff[atoms[i]][0]+ff[atoms[j]][0])/2
			eps_ij = np.sqrt(ff[atoms[i]][1]*ff[atoms[j]][1])
			C6s[i][j] = 4*eps_ij*sig_ij**6
			C12s[i][j] = 4*eps_ij*sig_ij**12
	return C6s, C12s

def backup(fname,counter):
	exists = os.path.exists('{}'.format(fname))
	if exists ==  False:
		return fname
	else:
		fname = fname.split('#')[-1]
		fname = fname.split('_')[0]
		fname = '#'+fname+'_'+str(counter)
		counter+=1
		return backup(fname,counter)

def log(step,energies,vels,forces,dt):
	potential = np.sum(energies)/2
	vrms = np.sqrt(np.mean(np.einsum("ij, ij -> i", vels, vels)))
	f_mean = np.mean(np.einsum("ij, ij -> i", forces, forces))
	print('#####################################')
	print(f'Step: \t\t {step:>8d}')
	print(f'Time: \t\t {(dt*step)/1000:>8.3f} ps')
	print(f'Potential: \t {potential:>8.2f} kJ/mol')
	print(f'V_rms: \t\t {vrms*100:>8.3f} nm/ps') # convert A/fs to nm/ps
	print(f'F_mean: \t {f_mean:>8.3f} kJ/(mol A)')

def main():
	mdp = tl.read_mdp(args.mdp)
	ff = tl.read_itp(mdp['itp'])

	# Set simulation parameters
	out ='traj.pdb'
	dt = mdp['dt']
	dt2 = 0.5*dt
	dt3 = 0.5*dt**2
	nsteps = mdp['nsteps']
	save = mdp['save']
	T = mdp['T']
	vdw_cut = mdp['vdw-cut']
	ensemble = mdp['ensemble']

	print(mdp)

	N, atoms, poss, vels, box,masses = read_box(args.pdb,T,ff)

	print(f'Generated initial velocities for temperature {T}')

	C6s, C12s = generate_lj_params(atoms,ff)

	# Create a backup if traj.pdb exists
	new_name = backup(out,0)
	if out == new_name:
		os.system(f"touch {out}")
	else:
		print(f"Found a file with the same output name will back it up to {new_name}")
		os.system(f"mv {out} '{new_name}'")
		os.system(f"touch {out}")

	# Compute initial forces and save initial coordinates with the stats
	energies,forces,old_accs = compute_forces(poss,box,C6s,C12s,masses,vdw_cut)
	tl.write_pdb(out,'a',box,atoms,poss,0)
	log(0,energies,vels,forces,dt)

	# Main loop
	t1 = time.time()
	for step in range(1,nsteps+1):
		new_pos = update_positions(N,box,poss,vels,old_accs,dt,dt3)
		energies,forces,accs = compute_forces(new_pos,box,C6s,C12s,masses,vdw_cut)
		vels = update_velocities(N,vels,old_accs,accs,dt2)
		old_accs = accs
		if step%save == 0:
			tl.write_pdb(out,'a',box,atoms,new_pos,step)
			log(step,energies,vels,forces,dt)
	delta = (time.time()-t1)/86400
	simulated_time = (nsteps*dt)*1e-6
	speed = simulated_time/delta
	print('#####################################')
	print(f'Performace {speed:5.2f} ns/day')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Dynamics')
	parser.add_argument('pdb', type=str,help='pdb file')
	parser.add_argument('mdp',type=str,help='mdp file')
	args = parser.parse_args()
	main()
