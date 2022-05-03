
from prody import *
from simtk.unit import *
from simtk.openmm.app import *
from simtk.openmm import *
import numpy as np
from pdbfixer import PDBFixer
import mdtraj as mt
from mdtraj.reporters import NetCDFReporter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import scipy.stats as st
from openmmtools import testsystems, cache, alchemy
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.mcmc import *
from sklearn.metrics import *
import simtk.openmm.app.element as elem
from pymbar import MBAR, timeseries
import math
import torch as torch
import pyemma
from pyemma.util.contexts import settings


def diagonalize(matrix):
    matrix_tc = torch.from_numpy(matrix)
    if torch.cuda.is_available():
        matrix_tc = matrix_tc.to(device=torch.device("cuda"))
    eig = torch.symeig(matrix_tc).eigenvalues
    eig = eig.to(device=torch.device("cpu"))
    eig = eig.numpy()
    eig = eig[np.argsort(eig)]
    eig = eig[6:]
    return eig

def annealingCycle(simulation: Simulation, initial_temperature=300 * kelvin, max_temperature=500 * kelvin,
                   min_temperature=300 * kelvin, steps_rising=100, steps_falling=1000, step_for_window=10, step_at_max=1000):
    """
        perform an Annealing on a Simulation from initial_temperature to max_temperature and fall to min_temperature in a total of total_steps
    """
    integrator: LangevinIntegrator = simulation.integrator

    if steps_rising > 0:
        a = (max_temperature - initial_temperature) / steps_rising

        for i in (range(steps_rising)):
            integrator.setTemperature(initial_temperature + a * i / steps_rising)
            simulation.step(step_for_window)
    print("rising ")
    integrator.setTemperature(max_temperature)
    if step_at_max > 0:
        simulation.step(step_at_max)
    print("top ")
    a = (min_temperature - max_temperature) / steps_falling

    for i in (range(steps_falling)):
        integrator.setTemperature(max_temperature + a * i / steps_falling)
        simulation.step(step_for_window)
    print("down")

def coolDown(simulation: Simulation, initial_temperature=300 * kelvin, min_temperature= 0 * kelvin, steps_falling=1000, step_for_window=10, step_at_max=1000):
    """
        perform an Annealing on a Simulation from initial_temperature to max_temperature and fall to min_temperature in a total of total_steps
    """
    integrator: LangevinIntegrator = simulation.integrator

    if step_at_max > 0:
        simulation.step(step_at_max)
    print("top ")
    a = (initial_temperature - min_temperature) / steps_falling

    for i in (range(steps_falling)):
        integrator.setTemperature(initial_temperature - a * i / steps_falling)
        simulation.step(step_for_window)
    print("down")

def minimizationCycle(simulation: Simulation, MiniTolerance=0.1, MaxMiniCycle=1000, NumMiniStepPerCycle=100, MiniEnergyExchangeRatio=1e-3):
    """
        Designed energy minimization cycle to minimize the structure such that the system mean force would fall around 2e-07 kcal/(A mol).
        This function will use the default positions to minimize.
        If the user did not to self.CPUPreMinimization() first, then the initial input positions will be used.
        Otherwise the pre-minimized positions will be used for performing the minimization cycle in CUDA platform.
        Args:
        MiniTolerance (energy=0*kilojoule/mole): The energy tolerance to which the system should be minimized set for each cycle.
        MaxMiniCycle (int=1000): The maximum number of cycles to perform energy minimizations.
        NumMiniStepPerCycle (int=10000): MaxIterations for each cycle of energy minimization.
        MiniForceRatio (double=1e-6): The order of mean force that the minimization cycle should eliminated.
    """

    PreMinimizedState: State = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
    PreMinimizedForces = PreMinimizedState.getForces(asNumpy=True).value_in_unit(kilocalorie_per_mole / angstrom)
    PreMinimizedMeanForce = np.linalg.norm(PreMinimizedForces, axis=1).mean() * (kilocalorie_per_mole / angstrom)
    prec_energy = PreMinimizedState.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)

    for i in range(MaxMiniCycle):

        simulation.minimizeEnergy(tolerance=MiniTolerance * kilocalorie_per_mole, maxIterations=NumMiniStepPerCycle)
        currentState: State = simulation.context.getState(getEnergy=True)
        current_Mean_Energy = currentState.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
        ratio = abs((current_Mean_Energy - prec_energy)/prec_energy)
        #print (ratio)
        if ratio < MiniEnergyExchangeRatio:
            print("Minimum reached. Exchange ratio: ", ratio)
            break
        prec_energy = current_Mean_Energy
        if i == MaxMiniCycle - 1:
            print("Minimum not reached. Too few iterations: ", ratio)

 #       progress( MiniEnergyExchangeRatio, ratio, suffix='')


def range_ij (size_i, size_j):
    res = []
    for i in range(size_i):
        for j in range(size_j):
            res.append([i,j])
    return res

def get_binding_energy(spectrum_bound=None, spectrum_unbound=None, sort=True, use_n = -1, min_eigenval=0.0, mode=None):
    """get_binding_energy
    """
    deltaF = None
    if spectrum_unbound is not None and spectrum_bound is not None:

        if use_n == -1:
            size = min(spectrum_bound.size, spectrum_unbound.size)
        else:
            size = min(use_n, min(spectrum_bound.size, spectrum_unbound.size))

        l_bound_up = spectrum_bound[np.argsort(spectrum_bound)]
        l_unbound_up = spectrum_unbound[np.argsort(spectrum_unbound)]
        l_bound_down = spectrum_bound[np.argsort(spectrum_bound)[::-1]]
        l_unbound_down = spectrum_unbound[np.argsort(spectrum_unbound)[::-1]]

        deltaF = 0

        max_l = max(np.max(l_bound_up), np.max(l_unbound_up))
        for i in range(size):
            if mode == 'fast':
                if ((l_bound_down[i] > min_eigenval * max_l) and (l_unbound_down[i] > min_eigenval * max_l)) :
                    deltaF += np.log(l_bound_down[i]/l_unbound_down[i])/2.0
            if mode == 'slow':
                if ((l_bound_up[i] < max_l * min_eigenval ) and (l_unbound_up[i] < min_eigenval * max_l)) :
                    deltaF += np.log(l_bound_up[i] / l_unbound_up[i]) / 2.0

        deltaF = np.real(deltaF)

        print('Binding free energy (in units of kT): ', deltaF)
    return deltaF

def compute_energy_state (simulation, inKT = True):
    state: State = simulation.context.getState(getEnergy=True)
    if inKT:
        energy = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole) * 1.689
        print("energy (kT): ", energy)
    else:
        energy = state.getPotentialEnergy().in_units_of(kilocalorie_per_mole)
        print("energy : ", energy)
    return energy

def report_simulation_pdb (simulation: Simulation, pdb_name):
    reporter = PDBReporter(pdb_name, 1000)
    reporter.report(simulation, simulation.context.getState(getPositions=True))

def set_box (pdb, factor=1.0):
    maxSize_x = (max((pos[0] for pos in pdb.positions)) - min((pos[0] for pos in pdb.positions))).value_in_unit(
        nanometers)
    maxSize_y = (max((pos[1] for pos in pdb.positions)) - min((pos[1] for pos in pdb.positions))).value_in_unit(
        nanometers)
    maxSize_z = (max((pos[2] for pos in pdb.positions)) - min((pos[2] for pos in pdb.positions))).value_in_unit(
        nanometers)

    vectors = (Vec3(maxSize_x*factor, 0, 0), Vec3(0, maxSize_y*factor, 0), Vec3(0, 0, maxSize_z*factor))
    pdb.topology.setPeriodicBoxVectors(vectors * nanometer)

def rename_res (filename, from_name, to_name):
    os.system("sed -i \'s/"+ from_name +"/"+ to_name +"/g\' "+ filename)

def top_to_itp (top_name, itp_name):
    itp_field = ("moleculetype", "bondtypes", "angletypes", "dihedraltypes", "atomtypes", "atoms", "bonds", "angles", "dihedrals", "pairs")
    not_field = ("#include", "#ifdef", "#endif")
    top = open(top_name, 'r')
    itp = open(itp_name, 'w')
    actual_field = ''
    for line in top:
        words = line.split()
        if '[' in words:
            actual_field = words[1]
        if len(words) == 2 and  (words[0] in not_field):
            actual_field = ''
        if actual_field in itp_field:
            itp.write(line)
    itp.close()

def read_data_from_file (filename, columns = None, flags=["@", ";", "#"] ):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            words = line.split()
            if columns is None :
                col = range(len(words))
            else:
                col = columns
            if not (words[0][0] in flags):
                data.append([float(words[i]) for  i in col])
    return np.array(data)

def remove_line (pdb_prefix, first_word, index=0):
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q']
    pdb = open(pdb_prefix + '.pdb', 'r')
    pdb_fix = open(pdb_prefix + '_temp.pdb', 'w')
    chain = 0
    for line in pdb:
        words = line.split()
        if len(words) > index:
            if words[index] != first_word:
                pdb_fix.write(line)
    pdb.close()
    pdb_fix.close()
    os.rename(pdb_prefix + "_temp.pdb", pdb_prefix + ".pdb")

def get_energy(simulation: Simulation):
    state: State= simulation.context.getState(getEnergy=True)
    #print('energy(kcal/mol): ', state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole))
    return state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole) * 1.689

def prepare_amber_complex (cmplx_prefix, clean_prefix, lig_prefix, ligand_name):

    #rename_res(clean_prefix + '.pdb', 'CL', 'Cl')
    #rename_res(clean_prefix + '.pdb', 'FAX', 'F  ')
    #rename_res(clean_prefix + '.pdb', 'FAY', 'F  ')
    #rename_res(clean_prefix + '.pdb', 'FCX', 'F  ')
    #rename_res(clean_prefix + '.pdb', 'BR', 'Br')
    #rename_res(protein_prefix + '.pdb', 'ZN', 'Zn')
    #rename_res(clean_prefix + '.pdb', 'FE2', 'FE ')
    #rename_res(protein_prefix + '.pdb', 'CA', 'Ca')
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q']
    pdb = open(clean_prefix + '.pdb', 'r')
    pdb_lig = open(lig_prefix + '.pdb', 'r')
    pdb_fix = open(cmplx_prefix + '_temp.pdb', 'w')

    residues = ('GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN',
                'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'HYS')# 'CL', 'ZN', 'BR', 'CA', 'FE')
    chain = 0
    found = False
    for line in pdb:
        words = line.split()
        remove = False
        if len(line) > 21:
            t = list(line)
            t[21] = alpha[chain]
            line = ''.join(t)

        if words[0] != 'END':
            pdb_fix.write(line)

        if len(words) > 3:
            if words[0] == 'TER':# or (words[0] == 'CONECT'):
                chain += 1
            if words[2] == 'OXT':
                chain += 1
    pdb_fix.write('TER\n')
    chain += 1

    for line in pdb_lig:
        if len(line) > 21:
            t = list(line)
            t[21] = alpha[chain]
            line = ''.join(t)
        words = line.split()
        if words[0] != 'REMARK':
            pdb_fix.write(line)

    pdb_fix.write('END\n')

    pdb_fix.close()
    pdb.close()
    pdb_lig.close()
    os.rename(cmplx_prefix + "_temp.pdb", cmplx_prefix + ".pdb")

def prepare_amber_clean (clean_prefix):
    rename_res(clean_prefix + '.pdb', 'CL', 'Cl')
    rename_res(clean_prefix + '.pdb', 'FAX', 'F  ')
    rename_res(clean_prefix + '.pdb', 'FAY', 'F  ')
    rename_res(clean_prefix + '.pdb', 'FCX', 'F  ')
    rename_res(clean_prefix + '.pdb', 'BR', 'Br')
    rename_res(clean_prefix + '.pdb', 'FE2', 'FE ')
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q']
    pdb = open(clean_prefix + '.pdb', 'r')
    pdb_clean = open(clean_prefix + '_temp.pdb', 'w')

    residues = ('GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN',
                'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'HYS')  # 'CL', 'ZN', 'BR', 'CA', 'FE')
    chain = 0
    found = False
    for line in pdb:
        words = line.split()
        remove = False
        if len(line) > 21:
            t = list(line)
            t[21] = alpha[chain]
            line = ''.join(t)
        if len(words) > 3:
            if not words[1].isalpha():
                if words[3] in residues:
                    pdb_clean.write(line)
                elif (words[0] == 'TER') and (words[2] in residues):  # or (words[0] == 'CONECT'):
                    chain += 1
                    pdb_clean.write(line)
                if words[2] == 'OXT':
                    chain += 1
            else:
                pdb_clean.write(line)

    pdb.close()
    pdb_clean.close()
    os.rename(clean_prefix + "_temp.pdb", clean_prefix + ".pdb")

def enumerate_atom (file, atom_name):
    pdb = open(file + '.pdb', 'r')
    pdb_fix = open(file + '_temp.pdb', 'w')
    count = 0
    for line in pdb:
        if len(line) > 21:
            t = list(line)
            pos = str.find(line, atom_name)
            if pos > -1:
                count += 1
                t[pos + len(atom_name)] = str(count)
            line = ''.join(t)
            pdb_fix.write(line)
    pdb.close()
    pdb_fix.close()
    os.rename(file + '_temp.pdb', file + '.pdb')

def get_geometric_center (selected: Selection):
    pos = selected.getCoords()
    return [np.mean(pos[:, i]) for i in range(3)]

def get_n_waters (name, folder=''):
    leap = open(folder+"leap.log", 'r')
    found_solv = False
    n_waters = 0
    for line in leap:
        words = line.split()
        if len(words) > 1:
            if words[1] == "solvateBox" or words[1] == "solvatebox":
                if words[2] == name:
                    found_solv = True
                else:
                    found_solv = False
            if found_solv:
                if words[0] == "Added" or words[0] == "added":
                    n_waters = float(words[1])
                    found_solv = False
    leap.close()
    return  n_waters

def prepare_complex (lig_smile,pdb_name, pro_prefix, cln_prefix, lig_prefix, lig_name='LIG', folder='input/', solvate = False, generate_gro_files = False, residues_water_cap = None, keep_ions = True):
    ions_water = 0.0027
    path_mlg = '/home/giacomo/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
    if isinstance(lig_smile, str):
        lig_mol = Chem.MolFromSmiles(lig_smile)
        if lig_mol is None:
            lig_mol = Chem.MolFromInchi(lig_smile)
        lig_mol = Chem.AddHs(lig_mol)
        AllChem.EmbedMolecule(lig_mol)
    else:
        lig_mol = lig_smile
    Chem.MolToPDBFile(lig_mol, lig_prefix + '.pdb')
    rename_res(lig_prefix + '.pdb', 'UNL', 'LIG')

    positions = parsePDB(cln_prefix + '.pdb').getCoords()
    box_size = [max((pos[i] for pos in positions))-min((pos[i] for pos in positions)) for i in range(3)]
    box_center = [np.average([pos[i] for pos in positions]) for i in range(3)]
    num_modes = 9
    whole_selection = 'protein or (resname ' + lig_name + ')'
    os.system('rm leap.log')

    pdb = parsePDB(pdb_name)
    if keep_ions:
        writePDB(pro_prefix + "_clean_temp.pdb", pdb.select('protein or ion'))
    else:
        writePDB(pro_prefix + "_clean_temp.pdb", pdb.select('protein'))

    #fixer = PDBFixer(filename=pro_prefix + "_clean_temp.pdb")
    #PDBFile.writeFile(fixer.topology, fixer.positions, open(pro_prefix + "_clean_temp.pdb", 'w'), keepIds=True)

    os.system("pdb4amber -i " + pro_prefix + "_clean_temp.pdb -o " + pro_prefix + "_clean.pdb --nohyd --dry --add-missing-atoms\n")
    os.system('rm ' + pro_prefix + '_clean_*')
    #remove_line(pro_prefix + "_clean", "CONECT", 0)

    with open('script.com', 'w') as f:
        f.write(
            "open " + lig_prefix + ".pdb\n" +
            "addh spec :" + lig_name + "\n" +
            "write format pdb 0 " + lig_prefix + ".pdb\n" + "stop\n"
        )
    f = open('script.sh', 'w')
    f.write("#!/bin/bash\nchimera -n <<< read script.com\n")
    f.close()
    #os.system('bash script.sh')

    fixer = PDBFixer(filename=lig_prefix + ".pdb")
    PDBFile.writeFile(fixer.topology, fixer.positions, open(lig_prefix + ".pdb", 'w'), keepIds=True)

    rename_res(lig_prefix + '.pdb', 'CL', 'Cl')
    rename_res(lig_prefix + '.pdb', 'BR', 'Br')

    if not os.path.exists(lig_prefix + '.mol2'):
        os.system('antechamber -fi pdb -fo mol2 -i ' + lig_prefix + '.pdb -o ' + lig_prefix + '.mol2 -c bcc -nc ' + str(0) + ' -rn ' + lig_name + ' -ek "qm_theory=\'AM1\', grms_tol=0.05, scfconv=1.d-6, ndiis_attempts=700,"')
        if not os.path.exists('./' + lig_prefix + '.mol2'):
            os.system(
                'antechamber -fi pdb -fo mol2 -i ' + lig_prefix + '.pdb -o ' + lig_prefix + '.mol2 -c bcc -nc ' + str(1) + ' -rn ' + lig_name + ' -ek "qm_theory=\'AM1\', grms_tol=0.05, scfconv=1.d-6, ndiis_attempts=700,"')
    os.system('parmchk2 -i ' + lig_prefix + '.mol2 -o ' + lig_prefix + '.frcmod -f mol2')
    with open('leaprc.in', 'w') as f:
        f.write(
            "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
            "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n" +
            lig_name + " = loadmol2 " + lig_prefix + ".mol2\n" +
            "loadamberparams " + lig_prefix + ".frcmod\n" +
            "check " + lig_name + "\n"+
            "saveamberparm " + lig_name + " " + lig_prefix + ".prmtop " + lig_prefix + ".prmcrd\n")
        if solvate:
            f.write("solvateBox " + lig_name + " TIP3PBOX 10.0 \n" +
                    "addIonsRand " + lig_name + " K+ 0 \n" +
                    "addIonsRand " + lig_name + " Cl- 0 \n")

        #if not solvate and residues_water_cap is not None: TODO solvate cap: add water around some residues
        #    for res in residues_water_cap:
        #        f.write("solvateBox " + lig_name + " TIP3PBOX 10.0 \n" +
        #                "addIonsRand " + lig_name + " K+ 0 \n" +
        #                "addIonsRand " + lig_name + " Cl- 0 \n")
        f.write("quit\n")
    os.system('tleap -f leaprc.in')
    if solvate:
        n_waters_lig =  get_n_waters(lig_name)
        #n_waters_complex = get_n_waters("complex")

        with open('leaprc.in', 'w') as f:
            f.write(
                "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
                "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n" +
                lig_name + " = loadmol2 " + lig_prefix + ".mol2\n" +
                "loadamberparams " + lig_prefix + ".frcmod\n" +
                "check " + lig_name + "\n" +
                "solvateBox " + lig_name + " TIP3PBOX 10.0\n" +
                "addIonsRand " + lig_name + " K+  " + str(floor(n_waters_lig * ions_water)) + "\n" +
                "addIonsRand " + lig_name + " Cl- " + str(floor(n_waters_lig * ions_water)) + "\n" +
                "addIons " + lig_name + " K+  0\n" +
                "addIons " + lig_name + " Cl- 0\n" +
                "saveamberparm " + lig_name + " " + lig_prefix + "_w.prmtop " + lig_prefix + "_w.prmcrd\n" +
                "quit\n")
        os.system('tleap -f leaprc.in ')

    with open('script.com', 'w') as script:
        script.write('ligand = ' + lig_prefix + '.pdbqt\n'+
                     'receptor = ' + pro_prefix + '_clean.pdbqt\n' +
                     'center_x = ' + str(box_center[0]) + '\n' +
                     'center_y = ' + str(box_center[1]) + '\n' +
                     'center_z = ' + str(box_center[2]) + '\n' +
                     'size_x = ' + str(box_size[0]) + '\n' +
                     'size_y = ' + str(box_size[1]) + '\n' +
                     'size_z = ' + str(box_size[2]) + '\n' +
                     'out = ' + lig_prefix + '_vina.pdbqt\n' +
                     'num_modes = ' + str(num_modes) + '\n' +
                     'exhaustiveness = 30\n' +
                     'log = ' + folder + 'log.txt\n')

    with open('script.sh', 'w') as script:
        script.write('#!/bin/bash\n' +
                     'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                     '$pythonMGL ' + path_mlg + 'prepare_ligand4.py -l ' + lig_prefix + '.mol2 -A \'bonds\' -U \'lps\' -o ' + lig_prefix + '.pdbqt -C \n' +
                     '$pythonMGL ' + path_mlg + 'prepare_receptor4.py -r ' + pro_prefix + '_clean.pdb -o ' + pro_prefix + '_clean.pdbqt\n' )
    os.system('bash script.sh')
    rename_res(lig_prefix + '.pdbqt', 'Ho', 'HD')

    with open('script.sh', 'w') as script:
        script.write('#!/bin/bash\n' +
                     'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                     'vina --config script.com\n'+
                     'vina_split --input ' + lig_prefix + '_vina.pdbqt --ligand ' + lig_prefix + '_ \n')
    os.system('bash script.sh')

    modes = []
    dG_0 = 0
    for i in range(1, num_modes + 1):
        if os.path.exists(lig_prefix + '_' + str(i) + '.pdbqt'):
            dG, rmsd_l = read_log(folder + 'log.txt', line=str(i), col=[1, 2])
            if i == 1:
                modes.append(i)
                dG_0 = dG
                with open('script.sh', 'w') as script:
                    script.write('#!/bin/bash\n' +
                                 'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                                 '$pythonMGL ' + path_mlg + 'pdbqt_to_pdb.py -f ' + lig_prefix + '_' + str(i) + '.pdbqt\n')
                os.system('bash script.sh')
                enumerate_atom(lig_prefix + '_' + str(i), 'Cl')
                enumerate_atom(lig_prefix + '_' + str(i), 'Br')

            elif (np.fabs(dG - dG_0) < 0.25) and (rmsd_l >= 2.0) and (rmsd_l < 50):
                modes.append(i)
                with open('script.sh', 'w') as script:
                    script.write('#!/bin/bash\n' +
                                 'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                                 '$pythonMGL ' + path_mlg + 'pdbqt_to_pdb.py -f ' + lig_prefix + '_' + str(i) + '.pdbqt\n')
                os.system('bash script.sh')
                enumerate_atom(lig_prefix + '_' + str(i), 'Cl')
                enumerate_atom(lig_prefix + '_' + str(i), 'Br')
            else:
                os.system('rm ' + lig_prefix + '_' + str(i) + '.pdbqt')

    #TODO add ALL TYPES OF METALS
    #modes = [1]

    # create the complex system for the receptor and the ligand
    for i in modes:
        protein_prefix = pro_prefix + '_' + str(i)
        ligand_prefix =  lig_prefix + '_' + str(i)

        with open('script.com', 'w') as f:
            f.write(
                "open " + ligand_prefix + ".pdb\n" +
                "addh spec :" + lig_name + " \n" +
                "write format pdb 0 " + ligand_prefix + ".pdb\n"+"stop\n"
            )
        f = open('script.sh', 'w')
        f.write("#!/bin/bash\nchimera -n <<< read script.com\n")
        f.close()
        #os.system('bash script.sh')

        prepare_amber_complex(protein_prefix + '_temp', pro_prefix + '_clean', ligand_prefix, lig_name)

        fixer = PDBFixer(filename=protein_prefix + "_temp.pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(protein_prefix + "_temp.pdb", 'w'), keepIds=True)

        os.system(
            "pdb4amber -i " + protein_prefix + "_temp.pdb -o " + protein_prefix + ".pdb --nohyd --add-missing-atoms\n")
        os.system('rm ' + protein_prefix + '_*')
        remove_line(protein_prefix, "CONECT", 0)
        os.system('rm leap.log')
        pdb = parsePDB(protein_prefix + '.pdb')
        lig_center = get_geometric_center(pdb.select('(resname ' + lig_name + ')'))
        with open('leaprc.in', 'w') as f:
            f.write(
                "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
                "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n")
            if keep_ions:
                f.write("source /home/giacomo/amber18/dat/leap/cmd/leaprc.water.spce\n")
            f.write(
                lig_name + " = loadmol2 " + lig_prefix + ".mol2\n" +
                "loadamberparams " + lig_prefix + ".frcmod\n" +
                "complex = loadpdb " + protein_prefix + ".pdb\n"+
                "saveamberparm complex " + protein_prefix + ".prmtop " + protein_prefix + ".prmcrd\n")
            if solvate:
                f.write("solvateBox complex TIP3PBOX 10.0 \n" +
                        "addIonsRand complex K+ 0 \n" +
                        "addIonsRand complex Cl- 0 \n")
            if residues_water_cap is not None:
                f.write("solvateBox complex TIP3PBOX 10.0 \n" +
                        "savepdb complex " + protein_prefix + ".pdb\n")
            f.write("quit\n")
        os.system('tleap -f leaprc.in')

        if residues_water_cap is not None:
            pdb = parsePDB(protein_prefix + ".pdb")
            if keep_ions:
                select_string = 'protein or ion or (water and same residue as (within 5 of ('
            else:
                select_string = 'protein or (water and same residue as (within 5 of ('
            for res in residues_water_cap:
                select_string += str(res) + ' '
            select_string += ' ))) or resname ' + lig_name
            writePDB(protein_prefix + "_temp.pdb", pdb.select(select_string))
            os.system(
                "pdb4amber -i " + protein_prefix + "_temp.pdb -o " + protein_prefix + ".pdb --add-missing-atoms\n")
            os.system('rm ' + protein_prefix + '_temp*')

            with open('leaprc.in', 'w') as f:
                f.write(
                    "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
                    "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n")
                if keep_ions:
                    f.write("source /home/giacomo/amber18/dat/leap/cmd/leaprc.water.spce\n")
                f.write(
                    lig_name + " = loadmol2 " + lig_prefix + ".mol2\n" +
                    "loadamberparams " + lig_prefix + ".frcmod\n" +
                    "complex = loadpdb " + protein_prefix + ".pdb\n" +
                    "saveamberparm complex " + protein_prefix + ".prmtop " + protein_prefix + ".prmcrd\n")
                f.write("quit\n")
            os.system('tleap -f leaprc.in')

        if solvate:
            n_waters_complex = get_n_waters("complex")
            # n_waters_complex = get_n_waters("complex")

            with open('leaprc.in', 'w') as f:
                f.write(
                    "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
                    "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n")
                if keep_ions:
                    f.write("source /home/giacomo/amber18/dat/leap/cmd/leaprc.water.spce\n")
                f.write(
                    lig_name + " = loadmol2 " + lig_prefix + ".mol2\n" +
                    "loadamberparams " + lig_prefix + ".frcmod\n" +
                    "complex = loadpdb " + protein_prefix + ".pdb\n" +
                    "solvateBox complex TIP3PBOX 10.0 \n" +
                    "addIonsRand complex K+  " + str(floor(n_waters_complex * ions_water)) + "\n" +
                    "addIonsRand complex Cl- " + str(floor(n_waters_complex * ions_water)) + "\n" +
                    "addIons complex K+  0\n" +
                    "addIons complex Cl- 0\n" +
                    "saveamberparm complex " + protein_prefix + "_w.prmtop " + protein_prefix + "_w.prmcrd\n" +
                    "quit\n")
            os.system('tleap -f leaprc.in ')
            if generate_gro_files:
                os.system('amb2gro_top_gro.py -p ' + protein_prefix + '.prmtop -c ' + protein_prefix + '.prmcrd -t ' + protein_prefix + '.top -g ' + protein_prefix + '.gro -b ' + protein_prefix + '.pdb')
                gro = GromacsGroFile(protein_prefix + '.gro')
                top = GromacsTopFile(protein_prefix + '.top')

                bonds, angles, dihedrals, idx = get_costraints(top.topology, gro.getPositions(asNumpy=True), lig_name)

                with open(protein_prefix + '.top', 'a') as top_file:
                    top_file.write("[ intermolecular_interactions ]\n"+
                                   "[ bonds ]\n" +
                                   "; ai     aj    type   bA      kA     bB      kB\n"+
                                   str(idx[0][0]) + " " + str(idx[1][0]) + " 6 " + str(bonds[0]) + " 0.0 " + str(bonds[0]) +  " 4184.0\n" +
                                   "[ angles ]\n" +
                                   "; ai     aj    ak     type    thA      fcA        thB      fcB\n" +
                                   str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(idx[0][0]) + " 1 " + str(angles[0]) + " 0.0 " + str(angles[0]) + " 41.84\n" +
                                   str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(idx[0][1]) + " 1 " + str(angles[1]) + " 0.0 " + str(angles[1]) + " 41.84\n" +
                                   "[ dihedrals ]\n" +
                                   "; ai     aj    ak    al    type     thA      fcA       thB      fcB\n" +
                                   str(idx[1][2]) + " " + str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(idx[0][0]) + " 2 " + str(dihedrals[0]) + " 0.0 " + str(dihedrals[0]) + " 41.84\n" +
                                   str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(idx[0][1]) + " 2 " + str(dihedrals[1]) + " 0.0 " + str(dihedrals[1]) + " 41.84\n" +
                                   str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(idx[0][1]) + " " + str(idx[0][2]) + " 2 " + str(dihedrals[2]) + " 0.0 " + str(dihedrals[2]) + " 41.84\n"
                                   )

                #create gromacs input files
                os.system('amb2gro_top_gro.py -p ' + lig_prefix + '.prmtop -c ' + lig_prefix + '.prmcrd -t ' + lig_prefix + '.top -g ' + lig_prefix + '.gro -b ' + lig_prefix + '.pdb')
                os.system('amb2gro_top_gro.py -p ' + cln_prefix + '.prmtop -c ' + cln_prefix + '.prmcrd -t ' + cln_prefix + '.top -g ' + cln_prefix + '.gro -b ' + cln_prefix + '.pdb')
        print('amber file created')


    return modes

def prepare_clean (pdb_name, cln_prefix, lig_name='LIG', folder='input/', solvate = False, generate_gro_files = False, residues_water_cap = None, keep_ions=True):
    ions_water = 0.0027
    path_mlg = '/home/giacomo/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
    os.system('rm leap.log')
    whole_selection = 'protein or (resname ' + lig_name + ')'

    pdb = parsePDB(pdb_name)
    if keep_ions:
        writePDB(pdb_name, pdb.select('protein or ion'))
    else:
        writePDB(pdb_name, pdb.select('protein'))

    #fixer = PDBFixer(filename=pdb_name)
    #fixer.findNonstandardResidues()
    #fixer.replaceNonstandardResidues()
    #PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_name, 'w'), keepIds=True)

    os.system("pdb4amber -i " + pdb_name + " -o " + cln_prefix + ".pdb --nohyd --dry --add-missing-atoms\n")
    os.system('rm ' + cln_prefix + '_*')
    remove_line(cln_prefix, "CONECT", 0)

    with open('leaprc.in', 'w') as f:
        f.write(
            "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
            "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n" )
        if keep_ions:
            f.write("source /home/giacomo/amber18/dat/leap/cmd/leaprc.water.spce\n")
        f.write(
            "clean = loadpdb " + cln_prefix + ".pdb\n"+
            "saveamberparm clean " + cln_prefix + ".prmtop " + cln_prefix + ".prmcrd\n")
        if solvate:
            f.write("solvateBox clean TIP3PBOX 10.0 \n" +
                    "addIonsRand clean K+ 0 \n" +
                    "addIonsRand clean Cl- 0 \n")
        f.write("quit\n")
    os.system('tleap -f leaprc.in')
    if solvate:
        n_waters_clean = get_n_waters("clean")
        #n_waters_complex = get_n_waters("complex")
        print("WATERWATERWATERWATERWATERWATERWATERWATERWATERWATERWATER", n_waters_clean)
        with open('leaprc.in', 'w') as f:
            f.write(
                "source /home/giacomo/amber18/dat/leap/cmd/leaprc.gaff\n" +
                "source /home/giacomo/amber18/dat/leap/cmd/oldff/leaprc.ff99SBildn\n" )
            if keep_ions:
                f.write("source /home/giacomo/amber18/dat/leap/cmd/leaprc.water.spce\n")
            f.write(
                "clean = loadpdb " + cln_prefix + ".pdb\n" +
                "solvateBox clean TIP3PBOX 10.0 \n" +
                "addIonsRand clean K+  " + str(floor(n_waters_clean * ions_water * 0.8)) + "\n" +
                "addIonsRand clean Cl- " + str(floor(n_waters_clean * ions_water * 0.8)) + "\n" +
                "addIons clean K+  0\n" +
                "addIons clean Cl- 0\n" +
                "saveamberparm clean " + cln_prefix + "_w.prmtop " + cln_prefix + "_w.prmcrd\n" +
                "quit\n")
        os.system('tleap -f leaprc.in ')

        if generate_gro_files:
            #create gromacs input files
            os.system('amb2gro_top_gro.py -p ' + cln_prefix + '.prmtop -c ' + cln_prefix + '.prmcrd -t ' + cln_prefix + '.top -g ' + cln_prefix + '.gro -b ' + cln_prefix + '.pdb')
        print('amber file created')
    #os.system('rm leap.log')

def read_log (log_file, line = '1', col = 1):
    with open(log_file, 'r') as log:
        for lines in log:
            words = lines.split()
            if len(words) >= 3:
                if words[0] == line:
                    words = np.array([float(words[i]) for i in range(len(words))])
                    return words[col]

def get_metastable_states (prefix, top, files, k, dt, selection=None):
    positions_feat = pyemma.coordinates.featurizer(top)

    try:
        positions_feat.add_backbone_torsions (selstr='not resname LIG')
        positions_feat.add_sidechain_torsions(selstr='not resname LIG')
    except:
        positions_feat.add_all()
        print('error: positions_feat.add_...._torsions()')

    #if selection is not None:
    #    indxs = positions_feat.select(selection)
    #    contacts = []
    #    for i in range(len(indxs)):
    #        for j in range(i + 1, len(indxs)):
    #            contacts.append([indxs[i], indxs[j]])
    #    positions_feat.add_distances(contacts)
    #positions_feat.add_distances(positions_feat.pairs(positions_feat.select_Heavy()))

    positions_data = pyemma.coordinates.load(files, features=positions_feat)
    try:
        cluster = pyemma.coordinates.cluster_kmeans(positions_data, k=k   , max_iter=50, fixed_seed=1)
        dtrajs_concatenated = np.concatenate(cluster.dtrajs)
        print('k: ', k)
    except:
        print('error k')
        cluster = pyemma.coordinates.cluster_kmeans(positions_data, k=None, max_iter=50, fixed_seed=1)
        dtrajs_concatenated = np.concatenate(cluster.dtrajs)

    msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=5, dt_traj=dt)
    msm.pcca(len(msm.transition_matrix))
    print(len(msm.transition_matrix))
    positions_feat = pyemma.coordinates.featurizer(prefix + '_top.pdb')
    positions_feat.add_all()
    positions_source = pyemma.coordinates.source(files, features=positions_feat)
    pcca_samples = msm.sample_by_distributions(msm.metastable_distributions, 1)
    pyemma.coordinates.save_traj(positions_source, pcca_samples, outfile=prefix + '_samples.pdb')
    return prefix + '_samples.pdb'

def get_minimum (simulation: Simulation, prefix, n_frames, step_size, step_size_mcmc, positions, atom_selection=None, gamma =1.0, cutoff = 10.0, selection=None): # sampler: MCMCSampler,


    eigenvalues = []

    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(298.15)
    simulation.minimizeEnergy(maxIterations=5000, tolerance=0.5 * kilocalorie_per_mole)
    simulation.step(5000)

    pos = simulation.context.getState(getPositions=True).getPositions()
    report_simulation_pdb(simulation, prefix + '_top.pdb')
    for j in range(n_frames):
        simulation.reporters.clear()
        reporter_dcd = DCDReporter(prefix + '_' + str(j) + '.dcd', 500)
        simulation.reporters.append(reporter_dcd)
        simulation.context.setPositions(pos)
        simulation.context.setVelocitiesToTemperature(298.15)
        simulation.step(step_size)

    #TODO metastable states
    files = [prefix + '_' + str(j) + '.dcd' for j in range(n_frames)]

    simulation.reporters.clear()
    reporter_pdb = PDBReporter(prefix + '.pdb', 1000)
    #traj: mt.Trajectory = mt.load_pdb(get_metastable_states (prefix, prefix + '_top.pdb', files, int(0.8 * step_size / 500 ), str( 0.002 * 500) + ' ps', n_frames))
    name = get_metastable_states(prefix, prefix + '_top.pdb', files, int(sqrt(step_size * n_frames / 500)/2), str(0.002 * 500) + ' ps', selection=selection)
    remove_line(name.split('.')[0], "CRYST1", 0)
    file = PDBFile(name)

    for j in range(file.getNumFrames()):
        simulation.context.setPositions(file.getPositions(frame=j))
        simulation.minimizeEnergy(maxIterations=5000, tolerance=0.5 * kilocalorie_per_mole)
        reporter_pdb.report(simulation, simulation.context.getState(getPositions=True))

    pdb = parsePDB(prefix + '.pdb')
    eigenvalues_anm = []
    eigenvalues_gnm = []
    for j in range(pdb.numCoordsets()):
        if atom_selection is not None:
            pdb.setCoords(pdb.getCoordsets(j))
            atom_selected = pdb.select(atom_selection)
            anm = ANM(atom_selected)
            anm.buildHessian(atom_selected, gamma=gamma, cutoff=cutoff)
            #try:
            #    eigenvalues_anm.append(diagonalize(anm.getHessian()))
            #except:
            anm.calcModes(None)
            eigenvalues_anm.append(anm.getEigvals())

            gnm = GNM(atom_selected)
            gnm.buildKirchhoff(atom_selected, gamma=gamma, cutoff=cutoff)
            #try:
            #    eigenvalues_gnm.append(diagonalize(gnm.getKirchhoff()))
            #except:
            gnm.calcModes(None)
            eigenvalues_gnm.append(gnm.getEigvals())

    if atom_selection is not None:
        with open(prefix + '_ANM.eigv', 'w') as out:
            for i in range(len(eigenvalues_anm)):
                for j in range(len(eigenvalues_anm[i])):
                    out.write(str(j) + ' ' + str(eigenvalues_anm[i][j]) + '\n')

        with open(prefix + '_GNM.eigv', 'w') as out:
            for i in range(len(eigenvalues_gnm)):
                for j in range(len(eigenvalues_gnm[i])):
                    out.write(str(j) + ' ' + str(eigenvalues_gnm[i][j]) + '\n')

    return eigenvalues_anm, eigenvalues_gnm

def compute_rmsd(positions_0, positions):
    return sqrt(mean_squared_error(positions, positions_0))

def get_eigenvals_metastable_states (system: System, simulation: Simulation, positions, prefix, atom_selection=None, n_iterations_max = 1, n_iterations_min = 1, n_frames = 50, gamma= 1.0):
    thermodynamic_state = ThermodynamicState(system=system, temperature=298.15 * unit.kelvin)
    ghmc_move = GHMCMove(timestep=2.0 * unit.femtosecond, n_steps=500)
    langevin_move = LangevinDynamicsMove(timestep=2.0 * unit.femtosecond, n_steps=500)
    weighted_move = WeightedMove([(ghmc_move, 0.7), (langevin_move, 0.3)])
    sampler_state = SamplerState(positions=positions)
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=ghmc_move)

    out = open(prefix + '.rmsd','w')
    simulation.context.setPositions(positions)
    min_energy = 0
    min_frame = 0
    reporter = PDBReporter(prefix + '.pdb', 1000)
    eigenvalues=[]
    for j in range(n_frames):
        sampler.sampler_state.positions = positions
        sampler.minimize()
        sampler.run(n_iterations=np.random.randint(n_iterations_min, n_iterations_max))
        sampler.minimize(max_iterations=5000, tolerance=0.5*kilojoules_per_mole)

        simulation.context.setPositions(sampler.sampler_state.positions)
        energy = get_energy(simulation)
        rmsd = compute_rmsd(positions, sampler.sampler_state.positions)
        out.write(str(rmsd) + ' ' + str(energy) + '\n')
        reporter.report(simulation, simulation.context.getState(getPositions=True))
        if energy < min_energy:
            min_energy = energy
            min_frame = j


        print(j)
        if atom_selection is not None:
            pdb = parsePDB( prefix + '.pdb')
            pdb.setCoords(pdb.getCoordsets(j))
            atom_selected = pdb.select(atom_selection)
            anm = ANM(atom_selected)
            anm.buildHessian(atom_selected, gamma=gamma)
            anm.calcModes(n_modes=None)
            eigenvalues.append(anm.getEigvals())
    return eigenvalues
#
def get_vibrational_entropy (w):
   return np.sum([np.log(1-np.exp(-np.sqrt(w[i]) * 1.0e-8/(2 * np.pi * 417) )) - np.sqrt(w[i]) * 1e-8/( 417 * 2 * np.pi * (np.exp(np.sqrt(w[i]) * 1.0e-8/(2 * np.pi * 417) ) -1)) for i in range(len(w))])

def get_stable_water(pdb_prefix, lig_name, residues, n_min, traj=None, topology=None):
    if (traj is not None) and (topology is not None):
        traject = mt.load(traj,top=topology)
    else:
        traject = mt.load_pdb(pdb_prefix + '.pdb')

    h_bonds = mt.wernet_nilsson(traject, exclude_water=False)
    waters_indexes = []
    for bond in h_bonds:
        if traject.top.atom(bond[0]).residue.name == 'HOH':
            waters_indexes.append(traject.top.atom(bond[0]).index)
        if traject.top.atom(bond[1]).residue.name == 'HOH':
            waters_indexes.append(traject.top.atom(bond[1]).index)

    select = 'water and same residues as (within 6 of ( '
    for res in residues:
        select += res + ' '
    select += ' )) and index'
    for idx in waters_indexes:
        select += str(idx) + ' '

    sel = parsePDB(pdb_prefix + '.pdb').select(select)
    return sel.getIndices()
    #h_bonds = [h_bonds[i] for i in range(len(h_bonds)) if traject.topology.atom(h_bonds[i]) != ]



def draw_smiles (name, smiles, legends):

    molecules = []
    for smile in smiles:
        try:
            lig_mol = Chem.MolFromSmiles(smile)
            lig_mol = Chem.AddHs(lig_mol)
            AllChem.Compute2DCoords(lig_mol)
            molecules.append(lig_mol)
            print(smile)
        except:
            pass
    img = Draw.MolsToGridImage(molecules, molsPerRow=4, subImgSize=(480, 480), legends=legends)
    img.save(name)

def distance (a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def get_costraints (topology: Topology, positions, lig_name):

    min_dist = 50
    print(center_lig)
    couple = []
    couple_indx = []
    for residue_pro in topology.residues():
        if residue_pro.name != lig_name:
            for atom_pro in residue_pro.atoms():
                for residue in topology.residues():
                    if residue.name == lig_name:
                        for atom_lig in residue.atoms():
                            dist = distance(positions[atom_pro.index], positions[atom_lig.index])
                            if (dist < min_dist) and (atom_pro.index != atom_lig.index) and (atom_pro.element != elem.hydrogen) and (atom_lig.element != elem.hydrogen):
                                min_dist = dist
                                couple = [atom_pro, atom_lig]
                                couple_indx = [atom_pro.index, atom_lig.index]

    print(couple, min_dist)

    min_dist = 50
    min_indx = 0

    for bond_1 in topology.bonds():
        atoms_pro = []
        if bond_1[0].index == couple_indx[0]:
            atoms_pro.append(bond_1[0])
            atoms_pro.append(bond_1[1])

            for bond_2 in topology.bonds():
                if bond_2[0].index == bond_1[1].index and bond_2[0] != couple_indx[0]:
                    atoms_pro.append(bond_2[1])
                    break
                elif bond_2[1].index == bond_1[1].index and bond_2[1] != couple_indx[0]:
                    atoms_pro.append(bond_2[0])
                    break
            if len (atoms_pro) == 3:
                break

        elif bond_1[1].index == couple_indx[0]:
            atoms_pro.append(bond_1[1])
            atoms_pro.append(bond_1[0])

            for bond_2 in topology.bonds():
                if bond_2[0].index == bond_1[0].index and bond_2[0] != couple_indx[0]:
                    atoms_pro.append(bond_2[1])
                    break
                elif bond_2[1].index == bond_1[0].index and bond_2[1] != couple_indx[0]:
                    atoms_pro.append(bond_2[0])
                    break
            if len(atoms_pro) == 3:
                break


    print (atoms_pro[0], atoms_pro[1], atoms_pro[2])

    for bond_1 in topology.bonds():

        atoms_lig = []

        if bond_1[0].index == couple_indx[1]:
            atoms_lig.append(bond_1[0])
            atoms_lig.append(bond_1[1])

            for bond_2 in topology.bonds():
                if bond_2[0].index == bond_1[1].index and bond_2[0] != couple_indx[1]:
                    atoms_lig.append(bond_2[1])
                    break
                elif bond_2[1].index == bond_1[1].index and bond_2[1] != couple_indx[1]:
                    atoms_lig.append(bond_2[0])
                    break
            if len(atoms_lig) == 3:
                break

        elif bond_1[1].index == couple_indx[1]:
            atoms_lig.append(bond_1[1])
            atoms_lig.append(bond_1[0])

            for bond_2 in topology.bonds():
                if bond_2[0].index == bond_1[0].index and bond_2[0] != couple_indx[1]:
                    atoms_lig.append(bond_2[1])
                    break
                elif bond_2[1].index == bond_1[0].index and bond_2[1] != couple_indx[1]:
                    atoms_lig.append(bond_2[0])
                    break
            if len(atoms_lig) == 3:
                break

    print (atoms_lig)

    positions = positions
    dist = [calcDistance(positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index])]
    angles = [measure.getAngle(positions[atoms_pro[1].index], positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index]),
              measure.calcAngle(positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index], pos_lig[atoms_lig[1].index])]
    dihedral = [measure.getDihedral(positions[atoms_pro[2].index], positions[atoms_pro[1].index], positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index]),
                measure.getDihedral(positions[atoms_pro[1].index], positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index], pos_lig[atoms_lig[1].index]),
                measure.getDihedral(positions[atoms_pro[0].index], pos_lig[atoms_lig[0].index], pos_lig[atoms_lig[1].index], atoms_lig[atoms_indx_lig[2].index])]
    print (atoms_pro, indx)
    return dist, angle, dihedral, [[atoms_pro[i].index for i in range(len(atoms_pro))], [atoms_lig[i].index for i in range(len(atoms_lig))]]

def analyze_MBAR(u_kln, nstates, new_sample = None):

    N_k = np.zeros([nstates], np.int32)
    for k in range(nstates):
        [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k, k, :])
        indices = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)
        N_k[k] = len(indices)
        u_kln[k, :, 0:N_k[k]] = u_kln[k, :, indices].T
    mbar = MBAR(u_kln, N_k)
    DeltaF_ij, dDeltaF_ij, Theta_ij = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew', return_theta = True)
    return DeltaF_ij, dDeltaF_ij, Theta_ij

def run_lambda_path(lambda_path, system, topology, positions, n_steps_for_iteration, n_iterations):
    integrator = LangevinIntegrator(298.15 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(topology, system, integrator)
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    kT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * integrator.getTemperature()

    for k in range(len(lambda_path)):
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(298.15)
        simulation.context.setParameter(lambda_path[0][k], lambda_path[1][k])
        simulation.minimizeEnergy()

        for iteration in range(n_iterations):
            simulation.step(n_steps_for_iteration)
            print('state %5d iteration %5d / %5d' % (k, iteration, n_iterations))
            for l in range(len(lambda_path)):
                simulation.context.setParameter(lambda_path[0][l], lambda_path[1][l])
                u_kln[k, l, iteration] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kT

    DeltaF_ij, dDeltaF_ij, Theta_ij = analyze_MBAR(u_kln, len(lambda_path))
    print (DeltaF_ij, dDeltaF_ij, Theta_ij)
    return DeltaF_ij[len(DeltaF_ij)-1], dDeltaF_ij[len(dDeltaF_ij)-1], Theta_ij

def dF_costraints (r0, thA, thB):
    # ===================================================================================================
    # INPUTS
    # ===================================================================================================

    K = 8.314472 * 0.001  # Gas constant in kJ/mol/K
    V = 1.66  # standard volume in nm^3

    T = 298.15  # Temperature in Kelvin

    K_r = 4184.0  # force constant for distance (kJ/mol/nm^2)
    K_thA = 41.84  # force constant for angle (kJ/mol/rad^2)
    K_thB = 41.84  # force constant for angle (kJ/mol/rad^2)
    K_phiA = 41.84  # force constant for dihedral (kJ/mol/rad^2)
    K_phiB = 41.84  # force constant for dihedral (kJ/mol/rad^2)
    K_phiC = 41.84  # force constant for dihedral (kJ/mol/rad^2)

    # ===================================================================================================
    # BORESCH FORMULA
    # ===================================================================================================

    thA = math.radians(thA)  # convert angle from degrees to radians --> math.sin() wants radians
    thB = math.radians(thB)  # convert angle from degrees to radians --> math.sin() wants radians

    arg = (
            (8.0 * math.pi ** 2.0 * V) / (r0 ** 2.0 * math.sin(thA) * math.sin(thB))
            *
            (
                    ((K_r * K_thA * K_thB * K_phiA * K_phiB * K_phiC) ** 0.5) / ((2.0 * math.pi * K * T) ** (3.0))
            )
    )

    dG = - K * T * math.log(arg)

    print("dG_off = %8.3f kcal/mol" % (dG / 4.184))
    print("dG_on  = %8.3f kcal/mol" % (-dG / 4.184))

    return dG

def alchemical_trasformation(lig_name, lambda_path, lambda_path_solv, n_steps_for_iteration, n_iterations, prefix=None, system=None, topology=None,
                             prefix_solv=None, system_solv=None, topology_solv=None):
    if (system is None) or (topology is None):
        try:
            gro = GromacsGroFile(prefix + '.gro')
            top = GromacsTopFile(prefix + '.top', periodicBoxVectors=gro.getPeriodicBoxVectors())
            system: System = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
            system.addForce(MonteCarloBarostat(1.0, 298.15))
            topology = top.topology
            positions = gro.getPositions(asNumpy=True)
        except:
            prmtop = AmberPrmtopFile(prefix + '.prmtop')
            inpcrd = AmberInpcrdFile(prefix + '.prmcrd', loadBoxVectors=True)
            system: System = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
            system.addForce(MonteCarloBarostat(1.0, 298.15))
            topology = prmtop.topology
            positions = inpcrd.getPositions(asNumpy=True)

    if (system_solv is None) or (topology is None):
        try:
            gro = GromacsGroFile(prefix_solv + '.gro')
            top = GromacsTopFile(prefix_solv + '.top', periodicBoxVectors=gro.getPeriodicBoxVectors())
            system_solv: System = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
            system_solv.addForce(MonteCarloBarostat(1.0, 298.15))
            topology_solv = top.topology
            positions_solv = gro.getPositions(asNumpy=True)
        except:
            prmtop = AmberPrmtopFile(prefix_solv + '.prmtop')
            inpcrd = AmberInpcrdFile(system_solv + '.prmcrd', loadBoxVectors=True)
            system_solv: System = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds)
            system_solv.addForce(MonteCarloBarostat(1.0, 298.15))
            topology_solv = prmtop.topology
            positions_solv = inpcrd.getPositions(asNumpy=True)

    bonds, angles, dihedral, indx = get_costraints (topology, gro.getPositions(asNumpy=True), lig_name)
    bond_force_idx = 0
    angle_force_idx = 0
    dihedral_force_idx = 0
    extra_bond = []
    extra_angle = []
    extra_dihedral = []
    count = 0
    for force in system.getForces():
        if isinstance(force, HarmonicBondForce):
            extra_bond.append(force.addBond(indx[0][0], indx[1][0], bonds[0], 4184))
            bond_force_idx = count
        elif isinstance(force, HarmonicAngleForce):
            extra_angle.append(force.addAngle(indx[1][1], indx[1][0], indx[0][0], angles[0] * n.pi / 180.0, 41.84))
            extra_angle.append(force.addAngle(indx[1][0], indx[0][0], indx[0][1], angles[1] * n.pi / 180.0, 41.84))
            angle_force_idx = count
        elif isinstance(force, PeriodicTorsionForce):
            extra_dihedral.append(force.addTorsion(indx[1][2], indx[1][1], indx[1][0], indx[0][0], dihedral[0] * n.pi / 180.0, 41.84))
            extra_dihedral.append(force.addTorsion(indx[1][1], indx[1][0], indx[0][0], indx[0][1], dihedral[1] * n.pi / 180.0, 41.84))
            extra_dihedral.append(force.addTorsion(indx[1][0], indx[0][0], indx[0][1], indx[0][2], dihedral[1] * n.pi / 180.0, 41.84))
            dihedral_force_idx = count
        count +=1

    #TODO use only existing forces

    mdtraj_topo: mt.Topology = mt.Topology.from_openmm(topology)
    atoms_lig = mdtraj_topo.select('resname ' + lig_name)
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=atoms_lig, alchemical_bonds=extra_bond, alchemical_angles=extra_angle, alchemical_torsions=extra_dihedral)
    factory = alchemy.AbsoluteAlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(system, alchemical_region)

    DeltaF_i, dDeltaF_i, Theta_ij = run_lambda_path(lambda_path, alchemical_system, topology, positions, n_steps_for_iteration, n_iterations)
    DeltaF = np.sum(DeltaF_i)

    mdtraj_topo_solv: mt.Topology = mt.Topology.from_openmm(topology_solv)
    atoms_lig_solv = mdtraj_topo_solv.select('resname ' + lig_name)
    alchemical_region_solv = alchemy.AlchemicalRegion(alchemical_atoms=atoms_lig_solv)
    alchemical_system_solv = factory.create_alchemical_system(system_solv, alchemical_region_solv)

    DeltaF_i_solv, dDeltaF_i_solv, Theta_i_solv = run_lambda_path(lambda_path_solv, alchemical_system_solv, topology_solv, positions_solv, n_steps_for_iteration, n_iterations)
    DeltaF_solv = np.sum(DeltaF_i_solv)
    DeltaF_costraints = dF_costraints(bonds[0], angles[0], angles[1])

    return DeltaF - DeltaF_solv - DeltaF_costraints

def get_hydrogen_bonds (top, target_sel):
    return top.select('water and same residue as (within 3.2 of ( ' + target_sel + ' and element N O))) and element O')

def get_autodock (prefix, lig_name):
    path_mlg = '/home/giacomo/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
    pdb = parsePDB(prefix + '.pdb')
    dG=[]
    for i in range(pdb.numCoordsets()):
        writePDB(prefix + '_pro.pdb', pdb.select('protein'), csets = i)
        writePDB(prefix + '_lig.pdb', pdb.select('resname  ' + lig_name), csets=i)
        with open('script.sh', 'w') as script:
            script.write('#!/bin/bash\n' +
                         'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                         '$pythonMGL ' + path_mlg + 'prepare_ligand4.py -l '   + prefix + '_lig.pdb -o ' + prefix + '_lig.pdbqt\n' +
                         '$pythonMGL ' + path_mlg + 'prepare_receptor4.py -r ' + prefix + '_pro.pdb -o ' + prefix + '_pro.pdbqt\n')
        os.system('bash script.sh')
        rename_res(prefix + '_lig.pdbqt', 'Ho', 'HD')
        with open('script.sh', 'w') as script:
            script.write('$pythonMGL ' + path_mlg + 'compute_AutoDock41_score.py -l ' + prefix + '_lig.pdbqt -r ' + prefix + '_pro.pdbqt -o ' + prefix + '_score.eng -w \'w\'\n')
        os.system('bash script.sh')
        with open(prefix + '_score.eng', 'r') as inp:
            for line in inp:
                words = line.split()
                if len(words) > 2:
                    if (not words[2].isalpha()) and (words[2] != 'AutoDock4.1Score'):
                        dG.append(float(words[2]))
                        break
    return np.array(dG)

def get_autodock_score (prefix, lig_name):
    path_mlg = '/home/giacomo/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
    pdb = parsePDB(prefix + '.pdb')
    dG=[]
    for i in range(pdb.numCoordsets()):
        writePDB(prefix + '_pro.pdb', pdb.select('protein'), csets = i)
        writePDB(prefix + '_lig.pdb', pdb.select('resname  ' + lig_name), csets=i)
        with open('script.sh', 'w') as script:
            script.write('#!/bin/bash\n' +
                         'bash /home/giacomo/MGLTools-1.5.6/bin/mglenv.sh\n' +
                         '$pythonMGL ' + path_mlg + 'prepare_ligand4.py -l '   + prefix + '_lig.pdb -o ' + prefix + '_lig.pdbqt\n' +
                         '$pythonMGL ' + path_mlg + 'prepare_receptor4.py -r ' + prefix + '_pro.pdb -o ' + prefix + '_pro.pdbqt\n')
        os.system('bash script.sh')
        rename_res(prefix + '_lig.pdbqt', 'Ho', 'HD')
        with open('script.sh', 'w') as script:
            script.write('$pythonMGL ' + path_mlg + 'compute_AutoDock41_score.py -l ' + prefix + '_lig.pdbqt -r ' + prefix + '_pro.pdbqt -o ' + prefix + '_score.eng -w \'w\'\n')
        os.system('bash script.sh')
        with open(prefix + '_score.eng', 'r') as inp:
            for line in inp:
                words = line.split()
                if len(words) > 2:
                    if (not words[2].isalpha()) and (words[2] != 'AutoDock4.1Score'):
                        dG.append(float(words[2]))
                        break
    return np.array(dG)

def compute_dg (folder, eigenvalues_clean_anm, eigenvalues_cmplx_anm, eigenvalues_solv_anm, modes, use_n, min_eigenval, mode):
    min_dG = 0.0
    min_std = 0.0
    min_mod = 0

    for i in range(len(modes)):
        dG = []
        for l_cmplx in eigenvalues_cmplx_anm[i]:
            for l_clean in eigenvalues_clean_anm[i]:
                for l_solv in eigenvalues_solv_anm:
                    l_0 = np.append(l_clean, l_solv)
                    dG_ijk = -1.0 * (get_binding_energy(l_cmplx, l_0, use_n=use_n, min_eigenval=min_eigenval, mode=mode))
                    if (dG_ijk < 0):
                        print(dG_ijk)
                        dG.append(dG_ijk)

        std = np.std(dG)
        mean = np.mean(dG)
        print(mean, std)

        rej_dG = []
        for j in range(len(dG)):
            if np.fabs(dG[j] - mean) < 2 * std:
                rej_dG.append(dG[j])

        if len(rej_dG) > 0:
            std = np.std(rej_dG)
            mean = np.mean(rej_dG)

        if mean < min_dG:
            min_dG = mean
            min_mod = i
            min_std = std

    return min_dG, min_std, min_mod

def automataized_virtual_screening (protein_pdb, lig_smiles, n_steps_for_iteration = -1, n_iterations = -1,
                                    expected_dg = None, output_file=None, use_minimum_states = True, n_minimizations=10, solvate=False, folder='input/', use_n = -1, min_eigenval = 0.40,
                                    keep_best_n = 10, keep_ions = True, solvate_cap =False, include_hbonds=False, end_docking=False, start=0, open_mode='w'):
    prefix = protein_pdb.split('.')[0]
    soluteDielectric = 2.0
    solventDielectric = 78.5

    lig_name = 'LIG'
    protein_selection = 'protein'
    ligand_selection = 'resname LIG'
    whole_selection = '(' + protein_selection + ')' + ' or (' + ligand_selection + ')'
    clean_prefix = folder + prefix + "_clean"
    eigenvalues_clean_anm = []
    output = open(output_file, open_mode)
    output.close()
    try:
        pdb, header = parsePDB(folder + protein_pdb, header=True)
        assignSecstr(header, pdb)
        cutoff = 10.0
        gamma = GammaStructureBased(pdb)
    except:
        pdb = parsePDB(folder + protein_pdb, header=False)
        cutoff = 10.0
        gamma = 1.0

    prepare_clean(folder + protein_pdb, clean_prefix, lig_name='LIG', folder=folder, solvate=solvate, generate_gro_files=False,residues_water_cap=None, keep_ions=keep_ions)

    if solvate:
        prmtop_clean = AmberPrmtopFile(clean_prefix + '_w.prmtop')
        inpcrd_clean = AmberInpcrdFile(clean_prefix + '_w.prmcrd', loadBoxVectors=True)
        system_clean = prmtop_clean.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
                                                 constraints=HBonds)
        system_clean.addForce(MonteCarloBarostat(1.0,298.15))
    else:
        prmtop_clean = AmberPrmtopFile(clean_prefix + '.prmtop')
        inpcrd_clean = AmberInpcrdFile(clean_prefix + '.prmcrd', loadBoxVectors=True)
        system_clean = prmtop_clean.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                                 constraints=HBonds,
                                                 implicitSolvent=OBC2, soluteDielectric=soluteDielectric,
                                                 solventDielectric=solventDielectric,
                                                 implicitSolventSaltConc=0.15 * molar)
    system_clean.addForce(AndersenThermostat(298.15, 1.0))
    integrator_clean = VerletIntegrator(0.002 * picoseconds)
    simulation_clean = Simulation(prmtop_clean.topology, system_clean, integrator_clean)
    if inpcrd_clean.boxVectors is not None:
        simulation_clean.context.setPeriodicBoxVectors(*inpcrd_clean.boxVectors)
    simulation_clean.context.setPositions(inpcrd_clean.positions)

    if use_minimum_states:
        get_minimum(simulation_clean, clean_prefix, step_size=50000, step_size_mcmc=50,
                    n_frames=int(n_minimizations * 2), atom_selection=None, positions=inpcrd_clean.positions, gamma=gamma)
    else:
        get_eigenvals_metastable_states(system_clean, simulation_clean, inpcrd_clean.getPositions(asNumpy=True),
                                        clean_prefix, atom_selection=None, n_iterations_max=10,
                                        n_iterations_min=1, n_frames=int(n_minimizations * 2), gamma=gamma)

    best_dg = [[0, 0.0, 0,'']] * keep_best_n

    for k in range(start,len(lig_smiles)):
        try:
        #if True:
            output = open(output_file, 'a')
            os.system('mkdir ' + folder + prefix + '_' + str(k))
            complex_prefix = folder + prefix + '_' + str(k) + '/' + prefix + "_" + str(k)
            lig_prefix = folder + prefix + '_' + str(k) + '/' + prefix + '_LIG' + "_" + str(k)
            lig_smile = lig_smiles[k]
            protein_prefix_k = prefix + "_" + str(k) + '.pdb'
            residues_water_cap = None
            if solvate_cap:
                residues_water_cap = [' resname LIG']

            modes = prepare_complex(lig_smile, folder + protein_pdb, complex_prefix, clean_prefix, lig_prefix, lig_name, folder + prefix + '_' + str(k) + '/', solvate=solvate, residues_water_cap=residues_water_cap)

            if solvate:
                prmtop_solv = AmberPrmtopFile(lig_prefix + '_w.prmtop')
                inpcrd_solv = AmberInpcrdFile(lig_prefix + '_w.prmcrd', loadBoxVectors=True)
                system_solv = prmtop_solv.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
                                                       constraints=HBonds)
                system_solv.addForce(MonteCarloBarostat(1.0, 298.15))
            else:
                prmtop_solv = AmberPrmtopFile(lig_prefix + '.prmtop')
                inpcrd_solv = AmberInpcrdFile(lig_prefix + '.prmcrd', loadBoxVectors=True)
                system_solv = prmtop_solv.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                                       constraints=HBonds,
                                                       implicitSolvent=OBC2, soluteDielectric=5.0,
                                                       solventDielectric=solventDielectric,
                                                       implicitSolventSaltConc=0.15 * molar)
            print(system_solv.getDefaultPeriodicBoxVectors())
            system_solv.addForce(AndersenThermostat(298.15, 1.0))
            integrator_solv = VerletIntegrator(0.002 * picoseconds)
            simulation_solv = Simulation(prmtop_solv.topology, system_solv, integrator_solv)
            if inpcrd_solv.boxVectors is not None:
                simulation_solv.context.setPeriodicBoxVectors(*inpcrd_solv.boxVectors)
            simulation_solv.context.setPositions(inpcrd_solv.positions)

            eigenvalues_cmplx_anm = []
            eigenvalues_solv_anm = []
            eigenvalues_clean_anm = []
            eigenvalues_cmplx_gnm = []
            eigenvalues_solv_gnm = []
            eigenvalues_clean_gnm = []

            for i in modes:

                fixer = PDBFixer(filename=complex_prefix + '_' + str(i) + ".pdb")
                PDBFile.writeFile(fixer.topology, fixer.positions, open(complex_prefix + '_' + str(i) + ".pdb", 'w'), keepIds=True)
                pdb = parsePDB(complex_prefix + '_' + str(i) + '.pdb')

                res_num = pdb.select('((same residue as (within 4 of resname LIG)) and protein and noh)').getResindices()
                protein_selection_nma = '( protein and ( backbone or ( resindex '  # or ((same residue as (within 5 of resname LIG)) and protein and noh)'
                protein_selection_feature = '(backbone) or ( resi '
                count = 0
                prec_res = 0
                for res in res_num:
                    if (res != prec_res) or (count == 0):
                        protein_selection_nma += str(res) + ' '
                        protein_selection_feature += str(res) + ' '
                    prec_res = res
                    count += 1
                protein_selection_nma += ' )) and noh )'
                protein_selection_feature += ' ) and (mass > 1.5 ) )'
                if keep_ions:
                    protein_selection_nma += ' '

                if include_hbonds:
                    protein_selection_nma += ' or (water and same residue as (within 3.2 of ( resindex '
                    for res in res_num:
                       protein_selection_nma += str(res) + ' '
                    protein_selection_nma += ' and element N O)) and element O )'

                #print(protein_selection_nma)
                #protein_selection_nma = '( protein and noh ) or ion '
                print(protein_selection_nma)
                print(protein_selection_feature)


                ligand_selection_nma = 'resname ' + lig_name + ' and noh'
                ligand_selection_feature = 'resname ' + lig_name + ' and (mass > 1.5 )'
                whole_selection_nma = '(' + protein_selection_nma + ')' + ' or (' + ligand_selection_nma + ')'


                pdb = parsePDB(clean_prefix + '.pdb')
                eigenvalues_anm_temp = []
                eigenvalues_gnm_temp = []
                for j in range(pdb.numCoordsets()):
                    pdb.setCoords(pdb.getCoordsets(j))
                    atom_selected = pdb.select(protein_selection_nma)
                    anm = ANM(atom_selected)
                    anm.buildHessian(atom_selected, gamma=gamma, cutoff=cutoff)

                    #try:
                    #    eigenvalues_anm_temp.append(diagonalize(anm.getHessian()))
                    #except:
                    anm.calcModes(None)
                    eigenvalues_anm_temp.append(anm.getEigvals())

                    gnm = GNM(atom_selected)
                    gnm.buildKirchhoff(atom_selected, gamma=gamma, cutoff=cutoff)

                    #try:
                    #    eigenvalues_gnm_temp.append(diagonalize(gnm.getKirchhoff()))
                    #except:
                    gnm.calcModes(None)
                    eigenvalues_gnm_temp.append(gnm.getEigvals())

                with open(complex_prefix + '_' + str(i) + '_clean_ANM.eigv', 'w') as out:
                    for l in range(len(eigenvalues_anm_temp)):
                        for j in range(len(eigenvalues_anm_temp[l])):
                            out.write(str(j) + ' ' + str(eigenvalues_anm_temp[l][j]) + '\n')

                with open(complex_prefix + '_' + str(i)  + '_clean_GNM.eigv', 'w') as out:
                    for l in range(len(eigenvalues_gnm_temp)):
                        for j in range(len(eigenvalues_gnm_temp[l])):
                            out.write(str(j) + ' ' + str(eigenvalues_gnm_temp[l][j]) + '\n')

                eigenvalues_clean_anm.append(eigenvalues_anm_temp)
                eigenvalues_clean_gnm.append(eigenvalues_gnm_temp)


                if solvate:
                    prmtop_cmplx = AmberPrmtopFile(complex_prefix + '_' + str(i) + '_w.prmtop')
                    inpcrd_cmplx = AmberInpcrdFile(complex_prefix + '_' + str(i) + '_w.prmcrd', loadBoxVectors=True)
                    system_cmplx = prmtop_cmplx.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
                                                           constraints=HBonds)
                    system_cmplx.addForce(MonteCarloBarostat(1.0, 298.15))
                else:
                    prmtop_cmplx = AmberPrmtopFile(complex_prefix + '_' + str(i) + '.prmtop')
                    inpcrd_cmplx = AmberInpcrdFile(complex_prefix + '_' + str(i) + '.prmcrd', loadBoxVectors=True)
                    system_cmplx = prmtop_cmplx.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                                             constraints=HBonds,
                                                             implicitSolvent=OBC2, soluteDielectric=soluteDielectric,
                                                             solventDielectric=solventDielectric,
                                                             implicitSolventSaltConc=0.15 * molar)

                system_cmplx.addForce(AndersenThermostat(298.15, 1.0))
                integrator_cmplx = VerletIntegrator(0.002 * picoseconds)
                simulation_cmplx = Simulation(prmtop_cmplx.topology, system_cmplx, integrator_cmplx)
                if inpcrd_cmplx.boxVectors is not None:
                    simulation_cmplx.context.setPeriodicBoxVectors(*inpcrd_cmplx.boxVectors)
                simulation_cmplx.context.setPositions(inpcrd_cmplx.positions)
                if use_minimum_states:
                    l_anm, l_gnm = get_minimum(simulation_cmplx, complex_prefix + '_' + str(i), step_size=40000, step_size_mcmc=50, #10000
                                    n_frames=int(n_minimizations * 1), atom_selection=whole_selection_nma, positions=inpcrd_cmplx.positions, gamma=gamma, selection='( ' + protein_selection_feature + ') or (' + ligand_selection_feature +')')
                    eigenvalues_cmplx_anm.append(l_anm)
                    eigenvalues_cmplx_gnm.append(l_gnm)


                else:
                    l = get_eigenvals_metastable_states(system_cmplx, simulation_cmplx, inpcrd_cmplx.getPositions(asNumpy=True),
                                                    atom_selection=whole_selection_nma, prefix=complex_prefix + '_' + str(i),
                                                    n_iterations_max=10, n_iterations_min=1, n_frames=n_minimizations, gamma=gamma)
                    eigenvalues_cmplx_anm.append(l)

            if use_minimum_states:
                eigenvalues_solv_anm, eigenvalues_solv_gnm = get_minimum(simulation_solv, lig_prefix, step_size=10000, step_size_mcmc=50,
                                    n_frames=int(n_minimizations * 1), atom_selection=ligand_selection_nma, positions=inpcrd_solv.positions, gamma=gamma)
            else:
                eigenvalues_solv_anm = get_eigenvals_metastable_states(system_solv, simulation_solv,
                                                               inpcrd_solv.getPositions(asNumpy=True),atom_selection=ligand_selection_nma,
                                                               prefix=lig_prefix, n_iterations_max=10,
                                                               n_iterations_min=1, n_frames=n_minimizations, gamma=gamma)

            min_dG_anm, min_std_anm, min_mod_anm = compute_dg(folder, eigenvalues_clean_anm, eigenvalues_cmplx_anm, eigenvalues_solv_anm, modes, use_n, min_eigenval, mode='fast')
            min_dG_gnm, min_std_gnm, min_mod_gnm = compute_dg(folder, eigenvalues_clean_gnm, eigenvalues_cmplx_gnm, eigenvalues_solv_gnm, modes, use_n, min_eigenval, mode='slow')

            print(min_dG_anm, min_std_anm, min_mod_anm)
            print(min_dG_gnm, min_std_gnm, min_mod_gnm)

            dg_auto = read_log(folder + prefix + '_' + str(k) + '/log.txt') * 1.689

            if (expected_dg is not None) and (output is not None):
                output.write(
                    str(expected_dg[k]) + ' ' + str(min_dG_anm) + ' ' + str(min_std_anm) + ' ' + str(dg_auto) + ' ' + str(
                        (dg_auto + min_dG_anm)/2) + ' ' + str(min_dG_gnm) + ' ' + str(min_std_gnm) + '\n')
            else:
                output.write(protein_pdb + '_ligand_' + str(k) + ' ' + str(min_dG_anm) + ' ' + str(min_dG_gnm) + '\n')
            output.close()
        except:
            print('error')


def fast_screening(smiles, protein_pdb, folder, keep_best_n):
    prefix = protein_pdb.split('.')[0]

    lig_name = 'LIG'
    protein_selection = 'protein'
    ligand_selection = 'resname ' + lig_name
    whole_selection = '(' + protein_selection + ')' + ' or (' + ligand_selection + ')'
    clean_prefix = folder + prefix + "_clean"
    eigenvalues_clean_anm = []

    prepare_clean(folder + protein_pdb, clean_prefix, lig_name='LIG', folder=folder, solvate=solvate,
                  generate_gro_files=False, residues_water_cap=None, keep_ions=keep_ions)

    best_dg = [[0, 0.0, 0, '']] * keep_best_n

    dg = []
    computed_smiles =[]
    ligands = []
    k = 0
    for i in range(start, len(lig_smiles)):
        try:
            output = open(output_file, 'a')
            os.system('mkdir ' + folder + prefix + '_' + str(k))
            complex_prefix = folder + prefix + '_' + str(k) + '/' + prefix + "_" + str(k)
            lig_prefix = folder + prefix + '_' + str(k) + '/' + prefix + '_LIG' + "_" + str(k)
            lig_smile = lig_smiles[k]
            protein_prefix_k = prefix + "_" + str(k) + '.pdb'
            residues_water_cap = None
            if solvate_cap:
                residues_water_cap = [' resname LIG']

            modes = prepare_complex(lig_smile, folder + protein_pdb, complex_prefix, clean_prefix, lig_prefix, lig_name,
                                    folder + prefix + '_' + str(k) + '/', solvate=solvate,
                                    residues_water_cap=residues_water_cap)

            mol = Molecule()
            mol.load_pdb(lig_prefix + '_1.pdb')
            ligands.append(mol)

            dg_auto = read_log(folder + prefix + '_' + str(k) + '/log.txt') * 1.689
            dg.append(dg_auto)
            computed_smiles[k]=[lig_smile, dg_auto]
            k += 1
        except:
            pass

    ligands = ligands[np.argsort(dg)][(len(dg)-keep_best_n):]

    return ligands
