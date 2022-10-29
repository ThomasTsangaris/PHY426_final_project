import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md
import mdtraj
import time
import shutup 

shutup.please() # mute warnings

start = time.time() # start time

def plot_global_parameter(y: np.array, true_val: float, md_label: str, ylabel: \
                          str, title: str) -> None:
    """Plot the global parameter represented by the array y per frame according 
    to the size of y labelled as md_label with ylabel and the title parameter 
    as the y-label and title of the plot respectively. A horizontal dotted 
    line is plotted on the same graph indicating the global parameter value 
    true_val.
    
    Preconditions: 
        - y is a 1-dimensional array representing a global parameter
        throughout a MD trajectory where each entry corresponds to a frame
        - true_val is the appropriate global size parameter from an 
        experimentally verified structure on the Protein Data Bank
    """
    plt.figure(figsize=(9,4)) # adjust figure size
    plt.plot(range(1, len(y) + 1), y, color = 'r', label = md_label)
    xlabel = 'Frame' # plotting per frame, timescale info is in labels
    plt.xlabel(xlabel, fontsize = 14)
    plt.ylabel(ylabel, fontsize = 14)
    plt.title(title, fontsize = 14)
    plt.xlim(left = 0) # only show frame 1 and onward
    true_val_label = 'PDB' # true value corresponds to value from PDB
    # horizontal line indicating the PDB value 
    plt.axhline(y = true_val, color = 'k', linestyle = '--', linewidth = 1, \
                label = true_val_label)
    plt.legend()
    plt.show()
    

# Back-calculate Global Size and Shape Parameters:
# insulin A chain from protein databank (PDB)
expected_pdb = 'C:/pdb_426/md/insulin_a_chain.pdb'
# md universe object corresponding to insulin A chain from PDB
u_pdb = md.Universe(expected_pdb)
# number of residues in protein/peptide of interest
N = len(u_pdb.residues)

# choose terminal atoms for end-to-end distance computation
N_terminus_pdb = u_pdb.select_atoms('resid 1 and name N')
C_terminus_pdb = u_pdb.select_atoms('resid ' + str(N) + ' and name C')

# compute radius of gyration of PDB insulin A chain
Rg_pdb = u_pdb.residues.radius_of_gyration()
# compute end-to-end distance of PDB insulin A chain
Ree_pdb = np.linalg.norm(N_terminus_pdb.positions - C_terminus_pdb.positions)
# compute asphericity of PDB insulin A chain
A_pdb = u_pdb.residues.asphericity()


# pdb file that defines topology of system for molecular dynamics trajectory
gro_file = 'C:/pdb_426/md/conf.gro'
# trajectory file that defines the molecular dynamics simulation
traj_file = 'C:/pdb_426/md/traj_comp.xtc'
# md universe object corresponding to molecular dynamics simulation
u_md = md.Universe(gro_file, traj_file)
# mdtraj trajectory object corresponding to molecular dynamics simulation
traj = mdtraj.load(traj_file, top = gro_file).remove_solvent()
# number of frames in trajectory
num_frames = traj.n_frames

# choose terminal atoms for end-to-end distance computation # TODO
N_terminus_md = u_md.select_atoms('resid 22 and name N') 
C_terminus_md = u_md.select_atoms('resid 42 and name C')
# note that the residue number above is dependent on the convention used within
# the program used to produce the initial extended structure (PyMOL) and does
# not represent the actual numbering of the peptide

# lists for global parameters
Rg_arr = []
Ree_arr = []
A_arr = []

for frame in u_md.trajectory: # loop through each frame
    # compute and append radius of gyration
    Rg_arr.append(u_md.residues[:N].radius_of_gyration())
    # compute and append end-to-end distance
    Ree_arr.append(np.linalg.norm(N_terminus_md.positions - C_terminus_md.positions))
    # compute and append asphericity
    A_arr.append(u_md.residues[:N].asphericity())

# convert global parameter lists to arrays
Rg_arr = np.array(Rg_arr) # radius of gyration array
Ree_arr = np.array(Ree_arr) # end-to-end distance array
A_arr = np.array(A_arr) # asphericity array


# Back-calculate Secondary Structure:
# list of fraction of helical content in each frame (entry in array) 
helical_frame_arr = []
# list of arrays where each inner array contains ss assignments per frame
dssp_residue_arr = []

i = 1
for frame in traj: # loop through each frame in MD trajectory
    # determine secondary structure code of entire frame
    dssp_code = mdtraj.compute_dssp(frame)[0] # parse for just protein/peptide
    # number of helical residues
    num_helix = list(dssp_code).count('H')
    # append fraction of helical residues
    helical_frame_arr.append(num_helix)
    # append dssp assignments
    dssp_residue_arr.append(dssp_code)
    
    if num_helix: # check and print if any frames have helical residues
        # positions of helical residues
        helical_pos = np.where(np.array(dssp_code) == 'H')[0]
        print("\nResidues participating in a helix found at frame " + str(i) + \
              " and they occur at the following residue numbers:")
        for res_num in helical_pos: # loop through each residue that forms a helix
            print("\n" + str(res_num + 1))
    i += 1 # increment index

# convert lists to numpy arrays
helical_frame_arr = np.array(helical_frame_arr)
dssp_residue_arr = np.array(dssp_residue_arr)

# secondary structure assignments for first and last frame by residue
dssp_first_frame = dssp_residue_arr[0, :]
dssp_last_frame = dssp_residue_arr[-1, :]

print("\nThe first frame has", str(list(dssp_first_frame).count('H')), "helical residues.")
print("\nThe last frame has", str(list(dssp_last_frame).count('H')), "helical residues.")

# analyze the last M frames for global parameters
M = 50

# mean of Radius of Gyration for last M frames of MD simulation
Rg_md_mean = np.mean(Rg_arr[num_frames-M:])
# standard deviation of Radius of Gyration for last M frames of MD simulation
Rg_md_std = np.std(Rg_arr[num_frames-M:])
# standard deviation of the mean of Radius of Gyration for last M frames of MD simulation
Rg_md_sdom = Rg_md_std / np.sqrt(len(Rg_arr[num_frames-M:]))

# mean of End-to-end Distance for last M frames of MD simulation
Ree_md_mean = np.mean(Ree_arr[num_frames-M:])
# standard deviation of End-to-end Distance for last M frames of MD simulation
Ree_md_std = np.std(Ree_arr[num_frames-M:])
# standard deviation of the mean of End-to-end Distance for last M frames of MD simulation
Ree_md_sdom = Ree_md_std / np.sqrt(len(Ree_arr[num_frames-M:]))

# mean of Asphericity for last M frames of MD simulation
A_md_mean = np.mean(A_arr[num_frames-M:])
# standard deviation of Asphericity for last M frames of MD simulation
A_md_std = np.std(A_arr[num_frames-M:])
# standard deviation of the mean of Asphericity for last M frames of MD simulation
A_md_sdom = A_md_std / np.sqrt(len(A_arr[num_frames-M:]))


# print global parameters of PDB structure
print("\nRadius of Gyration of Insulin's A Chain from the Protein Databank:", \
      str(round(Rg_pdb, 2)), "Angstroms")
print("\nEnd-to-end Distance of Insulin's A Chain from the Protein Databank:", \
      str(round(Ree_pdb, 2)), "Angstroms")
print("\nAsphericity of Insulin's A Chain from the Protein Databank:", \
      str(round(A_pdb, 3)))

# print global parameters of last frame
print("\nRadius of Gyration of Insulin's A Chain from the Last Frame in MD Simulation:", \
      str(round(Rg_arr[-1], 2)), "Angstroms")
print("\nEnd-to-end Distance of Insulin's A Chain from the Last Frame in MD Simulation:", \
      str(round(Ree_arr[-1], 2)), "Angstroms")
print("\nAsphericity of Insulin's A Chain from the Last Frame in MD Simulation:", \
      str(round(A_arr[-1], 3)))

# print global statistics of last M frames
print("\nMean Radius of Gyration of Insulin's A Chain from Last", str(M), "Frames:", \
      str(round(Rg_md_mean, 2)), "Angstroms")
print("\nStandard Deviation of the Mean for Radius of Gyration of Insulin's A Chain from Last", \
      str(M), "Frames:", \
      str(round(Rg_md_sdom, 2)), "Angstroms")
print("\nMean End-to-end Distance of Insulin's A Chain from Last", str(M), "Frames:", \
      str(round(Ree_md_mean, 2)), "Angstroms")
print("\nStandard Deviation of the Mean for End-to-end Distance of Insulin's A Chain from Last", \
      str(M), "Frames:", \
      str(round(Ree_md_sdom, 2)), "Angstroms")
print("\nMean Asphericity of Insulin's A Chain from Last", str(M), "Frames:", \
      str(round(A_md_mean, 3)), "Angstroms")
print("\nStandard Deviation of the Mean for Asphericity of Insulin's A Chain from Last", \
      str(M), "Frames:", \
      str(round(A_md_sdom, 3)), "Angstroms")


# Plotting:    
    
# define legend label, y-axis label and title of plots to be made

md_label = "MD 30ns Trajectory" # specifies timescale of MD simulation

# Radius of Gyration
Rg_ylabel = r"$R_g$ (Angstroms)"
Rg_title = "Radius of Gyration of Insulin's A Chain in a 30ns MD Simulation"

# End-to-end Distance
Ree_ylabel = r"$R_{ee}$ (Angstroms)"
Ree_title = "End-to-end Distance of Insulin's A Chain in a 30ns MD Simulation"

# Asphericity
A_ylabel = r"$A$"
A_title = "Asphericity of Insulin's A Chain in a 30ns MD Simulation"

# plot global parameters as a function of frame

# plot radius of gyration
plot_global_parameter(Rg_arr, Rg_pdb, md_label, Rg_ylabel, Rg_title)

# plot end-to-end distance
plot_global_parameter(Ree_arr, Ree_pdb, md_label, Ree_ylabel, Ree_title)

# plot asphericity
plot_global_parameter(A_arr, A_pdb, md_label, A_ylabel, A_title)
 

# plot helical residue number versus frame
plt.figure(figsize=(9,4)) # adjust figure size
plt.plot(range(1, num_frames + 1), helical_frame_arr, color = 'r')
plt.xlabel("Frame", fontsize = 14)
plt.ylabel("Number of Helical Residues", fontsize = 14)
plt.title("Number of Helices in Insulin's A Chain from a 30ns MD Simulation", \
          fontsize = 14)
plt.show()


# print how long the program took to run
print("\nProgram took %s seconds to finish" % round(time.time() - start, 1))
