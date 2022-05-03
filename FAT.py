from torchvision.transforms import Scale

from Utility import *
from Samplers import *
from simtk.unit import *
import pyemma
import mdtraj as md
import sklearn.decomposition as skdec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import time
import scipy.stats as stats
from sys import stdout

picoseconds = pico * second
nanometer = nano * meter


class FAT(Score):
    def __init__(self, output_file=None, solvate=False, folder='input/', keep_ions=True,
                 open_mode='w', n_threads=1, recompute=False):

        self.output_file = output_file
        self.solvate = solvate
        self.working_dir = folder
        self.keep_ions = keep_ions
        self.open_mode = open_mode
        self.path_mlg = os.environ.get("PATH_MLG")
        self.n_threads = n_threads
        self.recompute = recompute
        if self.path_mlg is None:
            self.path_mlg = '/home/' + os.environ.get(
                "USER") + '/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'

    @staticmethod
    def load_data_refined_set(file_idx, folder, from_line=0, excluded=[], included=None):
        complx_dict = {}

        def get_dg(k: str):
            w = k.split("=")[1]
            m = w[len(w) - 2:]
            dg = w[:len(w) - 2]
            if m == "mM":
                return np.log10(float(dg) * 1e-3)
            if m == "uM":
                return np.log10(float(dg) * 1e-6)
            if m == "nM":
                return np.log10(float(dg) * 1e-9)
            if m == "pM":
                return np.log10(float(dg) * 1e-12)
            return None

        with open(file_idx, "r") as index_file:
            n = 0
            for line in index_file:
                if n >= from_line and not line.startswith("#"):
                    words = line.split()
                    pdb_name = words[0].upper()
                    print(pdb_name)
                    pdb_file = os.path.join(folder, pdb_name + ".pdb")
                    dg = get_dg(words[3])
                    lig_name = words[6][1:len(words[6]) - 1]
                    if (dg is not None) and (len(pdb_name) == 4) and (len(lig_name) == 3) and (
                            pdb_name not in excluded) and (included is None or pdb_name in included):
                        complx_dict[pdb_name] = {"dg": dg,
                                                 "protein_file": os.path.join(folder, pdb_name.lower(),
                                                                              pdb_name.lower() + "_protein.pdb"),
                                                 "ligand": lig_name,
                                                 "ligand_file": os.path.join(folder, pdb_name.lower(),
                                                                             pdb_name.lower() + "_ligand.mol2")}
                n += 1
        return complx_dict

    def __generate_ligands(self, th_id, from_i, to_i, ligands, ligands_name, ligands_folder, out):
        if not os.path.exists("THREAD_" + str(th_id)):
            os.mkdir("THREAD_" + str(th_id))
        os.chdir("THREAD_" + str(th_id))
        print("running thread " + str(th_id))
        for i in range(from_i, to_i):
            print("generating ligand " + ligands_name[i])
            out[i] = Ligand(ligands[i], ligands_name[i], ligands_folder[i])
        os.chdir("../")
        os.system("rm -r THREAD_" + str(th_id))

    def compute(self, complex: Crystal, use_pca=False):
        soluteDielectric = 1.0
        solventDielectric = 78.5

        # ONLY PROTEIN
        prmtop_clean = AmberPrmtopFile(complex.protein.prmtop_file)
        inpcrd_clean = AmberInpcrdFile(complex.protein.prmcrd_file, loadBoxVectors=True)
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

        # ONLY LIGAND
        prmtop_solv = AmberPrmtopFile(complex.ligand.prmtop_file)
        inpcrd_solv = AmberInpcrdFile(complex.ligand.prmcrd_file, loadBoxVectors=True)
        system_solv = prmtop_solv.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                               nonbondedCutoff=1 * nanometer,
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

        # PROTEIN + LIGAND
        prmtop_cmplx = AmberPrmtopFile(complex.prefix + '.prmtop')
        inpcrd_cmplx = AmberInpcrdFile(complex.prefix + '.prmcrd', loadBoxVectors=True)
        system_cmplx = prmtop_cmplx.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                                 nonbondedCutoff=1 * nanometer,
                                                 constraints=HBonds,
                                                 implicitSolvent=OBC2,
                                                 soluteDielectric=soluteDielectric,
                                                 solventDielectric=solventDielectric,
                                                 implicitSolventSaltConc=0.15 * molar)

        system_cmplx.addForce(AndersenThermostat(298.15, 1.0))
        integrator_cmplx = VerletIntegrator(0.002 * picoseconds)
        simulation_cmplx = Simulation(prmtop_cmplx.topology, system_cmplx, integrator_cmplx)
        if inpcrd_cmplx.boxVectors is not None:
            simulation_cmplx.context.setPeriodicBoxVectors(*inpcrd_cmplx.boxVectors)
        simulation_cmplx.context.setPositions(inpcrd_cmplx.positions)

        pdb_complex = parsePDB(complex.prefix + ".pdb")

        atoms_idx = pdb_complex.select(
            '(protein and noh) and (backbone or same residue as (within {} of resname {}))'.format(
                10, complex.ligand.name)).getIndices()

        protein_selection_nma = '(index '
        for idx in atoms_idx:
            protein_selection_nma += str(idx) + ' '
        protein_selection_nma = "protein and noh and backbone"# += ' ) and (mass > 1.5 )'
        ligand_selection_nma = 'resname ' + complex.ligand.name + ' and noh'
        whole_selection_nma = '(' + protein_selection_nma + ')' + ' or (' + ligand_selection_nma + ')'

        if use_pca:
            complex.generate_pqr()
            eigval_clean, eng_clean = self.get_eigenvalues_PCA(simulation_clean, complex.protein.prefix, 5, 50000 * 1,
                                                               inpcrd_clean.positions, complex.working_dir,
                                                               protein_selection_nma, True)
            eigval_ligand, eng_ligand = self.get_eigenvalues_PCA(simulation_solv, complex.ligand.prefix, 5, 50000 * 1,
                                                                 inpcrd_solv.positions, complex.ligand.working_dir,
                                                                 ligand_selection_nma, True)
            eigval_complex, eng_cmpx = self.get_eigenvalues_PCA(simulation_cmplx, complex.prefix, 5, 50000 * 1,
                                                                inpcrd_cmplx.positions, complex.working_dir,
                                                                whole_selection_nma, True)
            dG_h = (eng_cmpx - (eng_clean + eng_ligand)) / 2.476
            dG_s = self.get_binding_energy(eigval_complex, np.append(eigval_clean, eigval_ligand), use_n=1000)

            print('Binding free energy (average energy) (in units of kT): ', dG_h)
            print('Binding free energy (eigenvalues) (in units of kT): ', dG_s)

            dG = dG_s + dG_h
            return dG,  dG_s, dG_h
        else:
            eigval_clean = self.get_eigenvalue_NMA(simulation_clean, complex.protein.prefix, protein_selection_nma)
            eigval_ligand = self.get_eigenvalue_NMA(simulation_solv, complex.ligand.prefix, ligand_selection_nma)
            eigval_complex = self.get_eigenvalue_NMA(simulation_cmplx, complex.prefix, whole_selection_nma)
            dG = self.get_binding_energy(eigval_complex, np.append(eigval_clean, eigval_ligand))
        print('Binding free energy (in units of kT): ', dG)
        return dG

    def report(self, line, m='a'):
        with open(self.output_file, m) as out:
            for c in line:
                out.write(str(c) + " ")
            out.write("\n")
            out.close()

    class exp_gamma(Gamma):
        def __init__(self, r0=3):
            super().__init__()
            self.r0_2 = r0 * r0

        def gamma(self, dist2, i, j):
            return exp(-dist2 / self.r0_2)

    def get_eigenvalue_NMA(self, mySimulation, prefix, selection):
        mySimulation.minimizeEnergy(maxIterations=10000, tolerance=0.001 * kilocalorie_per_mole)
        FAT.report_simulation_pdb(mySimulation, prefix + '_top.pdb')
        pdb = parsePDB(prefix + '_top.pdb')
        anm = ANM(pdb.select(selection))
        gamma = self.exp_gamma(3)
        cutoff = 10
        anm.buildHessian(pdb.select(selection), gamma=gamma, cutoff=cutoff)
        t = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            hessian = torch.from_numpy(anm.getHessian()).float()
            hessian.requires_grad_(False)
            hessian.to(device)
            hessian = (hessian + hessian.transpose(0, 1)) / 2
            eig = hessian.symeig(eigenvectors=False)[0]
            eig = eig.cpu().detach().numpy()
        except:
            anm.calcModes(None)
            eig = anm.getEigvals()
        print("eigenvalues computed in: {:5f}".format(time.time() - t))
        # anm.calcModes(None)

        return eig

    @staticmethod
    def get_PCA(files, top, atom_indices=None, n_modes=None):
        traj_all = [md.load_dcd(file, top, atom_indices=atom_indices) for file in files]
        traj = md.join(traj_all)
        traj.superpose(traj, 0)
        pca = PCA()
        pca.buildCovariance(traj.xyz)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            covariance = torch.from_numpy(pca.getCovariance()).float()
            covariance.requires_grad_(False)
            covariance.to(device)
            covariance = (covariance + covariance.transpose(0, 1)) / 2
            eig = covariance.symeig(eigenvectors=False)[0]
            eig = eig.cpu().detach().numpy()
        except:
            eig = np.linalg.eigvalsh(pca.getCovariance())
        return eig

    def get_boxes_data(self, positions_cg, positions_fg):
        box_size_fg = tuple([(max([pos[i] for pos in positions_cg]) - min([pos[i] for pos in positions_cg])) * 1.1
                             for i in range(3)])
        center_coords_fg = tuple([np.average([pos[i] for pos in positions_fg]) for i in range(3)])

        box_size_cg = tuple(
            [(max(max((pos[i] for pos in positions_cg)), center_coords_fg[i] + box_size_fg[i] / 2) -
              min(min((pos[i] for pos in positions_cg)), center_coords_fg[i] - box_size_fg[i] / 2)) * 1.3
             for i in range(3)])

        center_coords_cg = tuple(
            [(max(max((pos[i] for pos in positions_cg)), center_coords_fg[i] + box_size_fg[i] / 2) +
              min(min((pos[i] for pos in positions_cg)), center_coords_fg[i] - box_size_fg[i] / 2)) * 0.5
             for i in range(3)])
        return box_size_fg, center_coords_fg, box_size_cg, center_coords_cg

    def compute_PBSA_energy(self, dcd_files, top, prefix, folder, dime=(65, 65, 65)):
        traj_all = [md.load_dcd(file, top) for file in dcd_files]
        traj: md.Trajectory = md.join(traj_all)
        traj.superpose(traj, 0)
        apbs = APBSUtility(folder)
        pqr = parsePQR(prefix + ".pqr")
        all_energies = []
        prmtop = AmberPrmtopFile(prefix + '.prmtop')
        inpcrd = AmberInpcrdFile(prefix + '.prmcrd', loadBoxVectors=True)
        system = prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds)

        system.addForce(AndersenThermostat(298.15, 1.0))
        integrator = VerletIntegrator(0.002 * picoseconds)
        simulation = Simulation(prmtop.topology, system, integrator)
        for i in range(traj.n_frames):
            pqr.setCoords(traj.xyz[i])
            writePQR(prefix + ".pqr", pqr)
            positions_cg = pqr.getCoords()
            positions_fg = pqr.getCoords()
            # compute the box size for Coarse Grain and Fine Grain
            box_size_fg, center_coords_fg, box_size_cg, center_coords_cg = self.get_boxes_data(positions_cg,
                                                                                               positions_fg)
            t = time.time()
            energy = apbs.run_APBS_mgauto(prefix + ".pqr", box_size_cg, box_size_fg, lpbe_npbe="lpbe", bcfl="sdh",
                                          chgm="spl4", srfm="spl4", center_coords_cg=center_coords_cg,
                                          center_coords_fg=center_coords_fg, dime=dime, suffix="")[2]
            print("energy computed in {:1.2f} s".format(time.time() - t))
            simulation.context.setPositions(traj.xyz[i])
            state: State = simulation.context.getState(getEnergy=True)
            all_energies.append(
                (energy + state.getPotentialEnergy()).in_units_of(kilojoule_per_mole).value_in_unit(kilojoule_per_mole))

        all_energies = np.array(all_energies)
        return np.mean(all_energies), np.std(all_energies)

    @staticmethod
    def find_equilibrium(energies_file, out_file=None):
        y = []
        k = 0
        with open(energies_file) as file:
            for line in file:
                words = line.split(",")
                if "#" not in words[0]:
                    eng = float(words[0])
                    y.append(eng)
                    k += 1
        y = np.array(y)

        sample_size = 250
        sample = [0] * sample_size
        t_sample = [0] * sample_size
        dEdt = []
        t = []
        for i in range(len(y)):
            if (i >= sample_size / 2) and (i <= len(y) - sample_size / 2):
                j = 0
                for k in range(int(i - sample_size / 2), int(i + sample_size / 2)):
                    sample[j] = y[k]
                    t_sample[j] = j
                    j += 1
                slope, intercept, r_value, p_value, std_err = stats.linregress(t_sample, sample)
                dEdt.append(slope)
                t.append(i)
        t = np.array(t)
        print(np.mean(dEdt), np.std(dEdt))
        rej_E_idx = []
        mean_dEdt = np.mean(dEdt)
        std_dEdt = np.std(dEdt)
        rejected = []
        for i in range(len(dEdt)):
            if np.abs(dEdt[i] - mean_dEdt) < std_dEdt:
                rej_E_idx.append(t[i])
            else:
                rejected.append(t[i])
        print(np.mean(y), np.std(y))
        if out_file is not None:
            with open(out_file, "w") as out:
                n_y = y[rej_E_idx]
                for e in range(len(n_y)):
                    out.write(str(e) + " " + str(n_y[e]) + "\n")
        print("average energy: " + str(np.mean(y[rej_E_idx])) + " +- " + str(np.std(y[rej_E_idx])) + "; N: " + str(
            len(rej_E_idx)))
        return np.mean(y[rej_E_idx]), np.std(y[rej_E_idx]), len(rej_E_idx)

    @staticmethod
    def get_average_energy(energies_files):
        y = []
        for energies_file in energies_files:
            with open(energies_file) as file:
                for line in file:
                    words = line.split(",")
                    if "#" not in words[0]:
                        eng = float(words[0])
                        y.append(eng)
        y = np.array(y)
        av_eng = np.mean(y)
        std_eng = np.std(y)

        return av_eng, std_eng

    def get_eigenvalues_PCA(self, simulation: Simulation, prefix, n_frames, step_size, positions, folder,
                            selection_str=None,
                            average_energy=True):

        simulation.context.setPositions(positions)
        self.report_simulation_pdb(simulation, prefix + '_top.pdb')
        nanosecond = nano * second
        for j in range(n_frames):
            simulation.reporters.clear()
            simulation.context.setPositions(positions)

            simulation.minimizeEnergy(maxIterations=5000, tolerance=0.5 * kilocalorie_per_mole)
            simulation.context.setVelocitiesToTemperature(298.15)
            t = time.time()
            # simulation.reporters.append(StateDataReporter(stdout, 5000, step=True, remainingTime=False,
            #                                              speed=True, totalSteps=step_size + 50000))
            print("starting equilibration")
            #if True:
            if not os.path.exists(prefix + '-' + str(j) + '.dcd') or self.recompute:
                simulation.step(50000)
                reporter_dcd = DCDReporter(prefix + '-' + str(j) + '.dcd', 5000)  # save each 10 picoseconds
                if average_energy:
                    simulation.reporters.append(StateDataReporter(prefix + '-' + str(j) + '.engy', 500, potentialEnergy=True))

                simulation.reporters.append(reporter_dcd)
                print("starting simulation")
                simulation.step(step_size)
            print("simulated {:1.2f} ns in {:1.2f} s".format((step_size * 0.002 * picoseconds).value_in_unit(nanosecond),
                                                             time.time() - t))

        if selection_str is not None:
            pdb = parsePDB(prefix + '_top.pdb')
            selected = pdb.select(selection_str)
            eigenvalues_anm = FAT.get_PCA([prefix + '-' + str(j) + '.dcd' for j in range(n_frames)],
                                          prefix + '_top.pdb', selected.getIndices())
        else:
            eigenvalues_anm = FAT.get_PCA([prefix + '-' + str(j) + '.dcd' for j in range(n_frames)],
                                          prefix + '_top.pdb')

        if average_energy:
            #av_eng, std_eng = self.compute_PBSA_energy([prefix + '-' + str(j) + '.dcd' for j in range(n_frames)],
            #                                           prefix + '_top.pdb',
            #                                           prefix,
            #                                           folder)
            av_eng, std_eng = self.get_average_energy([prefix + '-' + str(j) + '.engy' for j in range(n_frames)])
            print("energy in kJ/mol: {} +- {}".format(av_eng, std_eng))
            return eigenvalues_anm, av_eng
        else:
            return eigenvalues_anm

    @staticmethod
    def report_simulation_pdb(simulation: Simulation, pdb_name):
        reporter = PDBReporter(pdb_name, 1000)
        reporter.report(simulation, simulation.context.getState(getPositions=True))

    @staticmethod
    def get_binding_energy(spectrum_bound=None, spectrum_unbound=None, use_n=-1):
        """get_binding_energy
        """
        deltaF = None
        if spectrum_unbound is not None and spectrum_bound is not None:

            l_bound_down = spectrum_bound[np.argsort(spectrum_bound)[::-1]]
            l_unbound_down = spectrum_unbound[np.argsort(spectrum_unbound)[::-1]]
            if use_n > 0:
                size = min(l_unbound_down.size, l_bound_down.size, use_n)
            else:
                size = min(l_unbound_down.size, l_bound_down.size)

            deltaF = 0
            for i in range(size):
                if (l_bound_down[i] <= 0) or (l_unbound_down[i] <= 0):
                    break
                deltaF += np.log(l_bound_down[i] / l_unbound_down[i]) / 2.0

            deltaF = np.real(deltaF)
            print("Used {} / {} eigenvalues".format(i + 1, size))

        return -1.0 * abs(deltaF)
