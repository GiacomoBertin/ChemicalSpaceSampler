from builtins import staticmethod
from math import *
from Bio import pairwise2 as pw2
import numpy as np
from pdbfixer import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.openmm.app import element as elem
from rdkit.six import StringIO
from prody import *
from rdkit import Chem
from rdkit.Chem import AllChem
from simtk.unit import *
from rdkit.Chem import Draw
from multiprocessing import *
from scipy.spatial.transform import Rotation as R
from prody.measure import Transformation
from openbabel import openbabel as ob
from collections import namedtuple
from namedlist import namedlist
from vina import Vina
import os


class APBSUtility:
    def __init__(self, working_directory):
        self.working_directory = working_directory

    @staticmethod
    def __pdb2pqr(file_pdb, file_pqr, forcefield):
        os.system("pdb2pqr --whitespace --assign-only --ff={} {} {} ".format(forcefield, file_pdb, file_pqr))

    @staticmethod
    def pdb2pqr(file_pdb, file_pqr, forcefield="amber"):
        process = Process(target=APBSUtility.__pdb2pqr, args=(file_pdb, file_pqr, forcefield))
        process.start()
        process.join()
        lines = []
        n = 0
        vdw_dict = {}
        for i in range(ob.etab.GetNumberOfElements()):
            try:
                vdw_dict[ob.etab.GetSymbol(i)] = [ob.etab.GetVdwRad(i), ob.etab.GetMass(i)]
            except:
                pass
        try:
            ions = list(parsePDB(file_pdb).select("ion").iterAtoms())
        except:
            ions = None
        prec = ""
        m = 0
        with open(file_pqr, "r") as file:
            for line in file:
                if "ATOM" in line:
                    m += 1
                if not line.startswith("REMARK"):
                    new = list(line)
                    has_alpha = new[28:29] != " "
                    if has_alpha and prec != "".join(new[25:29]):
                        prec = "".join(new[25:29])
                        n += 1
                    if has_alpha:
                        new[28:29] = " "

                    new[25:28] = "{:3d}".format(n)
                    line = "".join(new)
                    lines.append(line)

        ions_charge = {"CA": +2.0, "ZN": +2.0, "FE": +1.0, "FE2": +2.0, "FE3": +3.0, "K": +1.0, "CL": -1.0,
                       "NA": +1.0, "AG": +1.0, "I": -1.0, "MG": +2.0, "BR": -1.0}
        if ions is not None:
            for atom in ions:
                m += 1

                lines.append("HETATM  {:4}  {:3}  {:3}     1     {: .3f}  {: .3f}  {: .3f}  {: .4f} {: .4f}\n".format(
                    m, atom.getName(), atom.getResname(), atom.getCoords()[0], atom.getCoords()[1], atom.getCoords()[2],
                    ions_charge[atom.getName()], vdw_dict["".join([c.lower() if i > 0 else c.upper()
                                                                   for i, c in enumerate(atom.getElement())])][0]))
        with open(file_pqr, "w") as file:
            file.writelines(lines)
        print("Exiting pdb2pqr")
        return

    @staticmethod
    def complex2pqr(protein_pqr, complex_pqr, ligand_pqr, name_residue="LIG"):
        lines = []
        with open(protein_pqr, "r") as file:
            for line in file:
                if not line.startswith("REMARK"):
                    lines.append(line)
        with open(ligand_pqr, "r") as file:
            for line in file:
                if not line.startswith("REMARK"):
                    lines.append(line)
        with open(complex_pqr, "w") as file:
            file.writelines(lines)

    @staticmethod
    def mol22pqr(file_mol2, file_prmtop=None, file_pqr=None, name_residue="LIG", file_pdb=None):
        atoms_dictionary = []
        vdw_dict = {}
        for i in range(ob.etab.GetNumberOfElements()):
            try:
                vdw_dict[ob.etab.GetSymbol(i)] = [ob.etab.GetVdwRad(i), ob.etab.GetMass(i)]
            except:
                pass
        if file_pdb is not None:
            atoms = parsePDB(file_pdb).iterAtoms()
            atoms_dict = {}
            for a in atoms:
                atoms_dict[a.getName()] = a.getElement()

        with open(file_mol2, "r") as f:
            while True:
                line = f.readline()
                if line.startswith("@<TRIPOS>ATOM"):
                    line = f.readline()
                    while not line.startswith("@<TRIPOS>BOND"):
                        words = line.split()
                        atom = words[1]
                        xyz = (float(words[2]), float(words[3]), float(words[4]))
                        ch = float(words[8])
                        if file_prmtop is not None:
                            atoms_dictionary.append({"charge": ch, "coords": xyz, "atom": atom})
                        else:
                            if file_pdb is None:
                                element = []
                                c = list(atom)
                                for l in c:
                                    if l.isnumeric():
                                        break
                                    else:
                                        element.append(l.lower())
                                if ''.join(element) in ["cb", "ca", "cc"]:
                                    element = list("CA")
                                element[0] = element[0].upper()
                                element = ''.join(element)
                            else:
                                element = atoms_dict[atom]

                            radii = vdw_dict[element][0]
                            atoms_dictionary.append({"charge": ch, "coords": xyz, "atom": atom, "radii": radii})

                        line = f.readline()
                        if f.tell() == os.fstat(f.fileno()).st_size:
                            break
                    break
                if f.tell() == os.fstat(f.fileno()).st_size:
                    break

        if file_prmtop is not None:
            with open(file_prmtop, "r") as f:
                while True:
                    line = f.readline()
                    if line.startswith("%FLAG RADII"):
                        line = f.readline()
                        line = f.readline()
                        idx = 0
                        while not line.startswith("%FLAG SCREEN"):

                            words = line.split()
                            for w in words:
                                radii = float(w)
                                atoms_dictionary[idx]["radii"] = radii
                                idx += 1
                            line = f.readline()
                            if f.tell() == os.fstat(f.fileno()).st_size:
                                break
                        break
                    if f.tell() == os.fstat(f.fileno()).st_size:
                        break

        with open(file_pqr, "w") as f:
            f.write("REMARK Powered by Giacomo\n")

            for i in range(len(atoms_dictionary)):  # HETATM
                f.write("HETATM {:4}   {:3}  {:3}     1     {: .3f}  {: .3f}  {: .3f}  {: .4f} {: .4f}\n".format(
                    i + 1, atoms_dictionary[i]["atom"], name_residue, atoms_dictionary[i]["coords"][0],
                    atoms_dictionary[i]["coords"][1], atoms_dictionary[i]["coords"][2], atoms_dictionary[i]["charge"],
                    atoms_dictionary[i]["radii"]
                ))
                # if i == (len(atoms_dictionary) - 1):
            f.write("TER    {:4}        {:3}     1\n".format(len(atoms_dictionary) + 1, name_residue))
            f.write("END\n")
        return

    @staticmethod
    def __run_APBS_mgauto(file_sh, directory, queue: Queue):
        user = os.getenv("USER")
        os.environ[
            "LD_LIBRARY_PATH"] = "/home/{}/APBS-1.5-linux64/lib:usr/lib/x86_64-linux-gnu:/usr/local/cuda-10.2/lib64".format(
            user)
        os.chdir(directory)
        res = os.popen("bash {}".format(file_sh)).read()
        queue.put(res)

    def read_output(self, lines):
        lines = lines.split("\n")
        for line in lines:
            if line.startswith("  Global net ELEC energy = "):
                return float(line.split()[len(line.split()) - 2]) * kilojoule_per_mole

    def run_APBS_mgauto(self, file_pqr, box_size_cg, box_size_fg, lpbe_npbe="lpbe", bcfl="sdh", chgm="spl4",
                        srfm="spl4", center_coords_cg=None, center_coords_fg=None, dime=(65, 65, 65), suffix=""):
        """
        The method starts by solving the equation on a coarse grid (i.e., few grid points) with large dimensions
        (i.e., grid lengths). The solution on this coarse grid is then used to set the Dirichlet boundary condition
        values for a smaller problem domain – and therefore a finer grid – surrounding the region of interest. The finer
        grid spacing in the smaller problem domain often provides greater accuracy in the solution.
        :param dime:              grid dimensions
        :param file_pqr:          starting PQR file
        :param box_size_cg:       size of the Coarse Grain box (used only for Dirichlet boundary condition)
        :param box_size_fg:       size of the Fine Grain box
        :param lpbe_npbe:         linearized or nonlinear Poisson-Boltzmann equation
        :param bcfl:              Specifies the type of boundary conditions used to solve the Poisson-Boltzmann equation
        :param chgm:              Specify the method by which the biomolecular point charges (i.e., Dirac delta functions)
                                  by which charges are mapped to the grid for a multigrid
        :param srfm:              Specify the model used to construct the dielectric and ion-accessibility coefficients
        :param center_coords_cg:  Center of the Coarse Grain box
        :param center_coords_fg:  Center of the Fine Grain box
        :return:
        """

        if center_coords_cg is None:
            center_cg = "1"
        else:
            center_cg = "{} {} {}".format(center_coords_cg[0], center_coords_cg[1], center_coords_cg[2])
        if center_coords_fg is None:
            center_fg = "1"
        else:
            center_fg = "{} {} {}".format(center_coords_fg[0], center_coords_fg[1], center_coords_fg[2])
        input_file = open(os.path.join(self.working_directory, "input.apbs"), "w")
        input_file.write("READ\n" +
                         "\tmol pqr {} \n".format(file_pqr) +
                         "END\n\nELEC\n" +
                         "\tmg-auto\n" +
                         "\tmol 1\n" +
                         "\t{}\n".format(lpbe_npbe) +
                         "\tbcfl {}\n".format(bcfl) +
                         "\tchgm {}\n".format(chgm) +
                         "\tsrfm {}\n".format(srfm) +
                         "\tswin {}\n".format(0.300) +
                         "\tcalcenergy total\n" +
                         "\tcalcforce no\n\n" +
                         # "\twrite pot dx {}\n".format(
                         #    os.path.join(self.working_directory, "grid_potentials.dx")) +
                         "\twrite charge dx {}\n".format(os.path.join(self.working_directory, "grid_charges")) +
                         "\twrite sspl dx {}\n\n".format(
                             os.path.join(self.working_directory, "grid_spline_solvent")) +
                         "\tcgcent {}\n".format(center_cg) +
                         "\tcglen {} {} {}\n".format(box_size_cg[0], box_size_cg[1], box_size_cg[2]) +
                         "\tfgcent {}\n".format(center_fg) +
                         "\tfglen {} {} {}\n".format(box_size_fg[0], box_size_fg[1], box_size_fg[2]) +
                         "\tdime {} {} {}\n".format(dime[0], dime[1], dime[2]) +
                         "\ttemp {}\n".format(298.150) +
                         "\tsrad {}\n".format(1.400) +
                         "\tsdie {}\n".format(78.570) +
                         "\tpdie {}\n".format(2.000) +
                         "\tion {} {} {}\n".format(1.000, 0.010, 2.000) +
                         "\tion {} {} {}\n".format(-1.000, 0.010, 2.000) +
                         "END\n\nPRINT\n" +
                         "\telecEnergy 1\n" +
                         "END\n\nQUIT\n"
                         )
        input_file.close()
        with open(os.path.join(self.working_directory, "run.sh"), "w") as sh_file:
            sh_file.write("#!/bin/bash\n" +
                          "cd {}\n".format(self.working_directory) +
                          "apbs {}\n".format(os.path.join(self.working_directory, "input.apbs")))
        queue = Queue()
        process = Process(target=APBSUtility.__run_APBS_mgauto,
                          args=(os.path.join(self.working_directory, "run.sh"), self.working_directory, queue,))
        process.start()
        res = queue.get()
        process.join()
        # res = os.popen("bash {}".format(os.path.join(self.working_directory, "run.sh"))).read()

        return os.path.join(self.working_directory, "grid_charges.dx"), \
               os.path.join(self.working_directory, "grid_spline_solvent.dx"), self.read_output(res)


class GROMACSUtility:

    @staticmethod
    def gmx_editconf(gro_input, gro_output, size=1.5, shape="cubic"):
        gro_input_prefix = gro_input.split(".gro")[0]
        gro_output_prefix = gro_output.split(".gro")[0]
        out = os.popen("gmx editconf -f " + gro_input_prefix + ".gro -o " + gro_output_prefix +
                       ".gro -bt " + shape + " -d " + str(size) + " -c ").read()
        return out

    @staticmethod
    def gmx_solvate(gro_input, top_input, gro_output):
        gro_input_prefix = gro_input.split(".gro")[0]
        gro_output_prefix = gro_output.split(".gro")[0]
        top_input_prefix = top_input.split(".gro")[0]
        out = os.popen("gmx solvate -cp " + gro_input_prefix + ".gro -cs spc216.gro -p " + top_input_prefix +
                       ".top -o " + gro_output_prefix + ".gro ").read()
        return out

    @staticmethod
    def gmx_grompp(gro_input, top_input, gro_output):
        # TODO
        gro_input_prefix = gro_input.split(".gro")[0]
        gro_output_prefix = gro_output.split(".gro")[0]
        top_input_prefix = top_input.split(".gro")[0]
        out = os.popen("gmx ").read()
        return out


class AmberUtility:
    # class AmberFiles:
    #    def __init__(self, prmtop, ):

    class Leap:
        def __init__(self, working_dir="./"):

            user = os.environ.get("USER")
            self.working_dir = working_dir
            self.lib_path = ""  # "/home/" + user + "/amber18/dat/leap/cmd/"
            self.command = ""
            self.variables = []

        def load_gaff(self):
            self.command += "source " + self.lib_path + "leaprc.gaff\n"

        def load_ff99SBildn(self):
            self.command += "source " + self.lib_path + "oldff/leaprc.ff99SBildn\n"

        def load_water_spce(self):
            self.command += "source " + self.lib_path + "leaprc.water.spce\n"

        def load_pdb(self, var_name, pdb_name):
            self.command += var_name + " = loadpdb " + pdb_name + ".pdb\n"
            self.variables.append(var_name)

        def load_mol2(self, var_name, mol2_name):
            self.command += var_name + " = loadmol2 " + mol2_name + ".mol2\n"
            self.variables.append(var_name)

        def load_amberparams(self, frcmod_name):
            self.command += "loadamberparams " + frcmod_name + ".frcmod\n"

        def check(self, var_name):
            self.command += "check " + var_name + "\n"

        def saveamberparm(self, var_name, output_name, formats=(".prmtop", ".prmcrd")):
            self.command += "saveamberparm " + var_name + " " + output_name + formats[0] + " " + output_name + formats[
                1] + "\n"

        def solvate(self, var_name, off=10.0, ions=("K+", "Cl-"), n_ions=(0, 0)):
            self.command += "solvateBox " + var_name + " TIP3PBOX " + str(
                off) + " \n" + "addIonsRand " + var_name + " " + ions[
                                0] + " " + str(n_ions[1]) + " \n" + "addIonsRand " + var_name + " " + ions[
                                1] + " " + str(n_ions[1]) + " \n"

        def addIons(self, var_name, ion, n=0):
            self.command += "addIons " + var_name + " " + ion + " " + str(n) + "\n"

        class AmberError(Exception):
            def __init__(self, message):
                self.message = message

            def __str__(self):
                return self.message

        def execute(self):
            with open(self.working_dir + '/leaprc.in', 'w') as f:
                f.write(self.command + "quit\n")
            # os.system('tleap -f ' + self.working_dir + '/leaprc.in')
            comm = os.popen('tleap -f ' + self.working_dir + '/leaprc.in').read()
            lines = comm.split("\n")
            for line in lines:
                if "Errors =" in line:
                    n_err = int(line.split("Errors =")[1][1])
                    if n_err > 0:
                        for l in lines:
                            print(l)
                        raise RuntimeError("Errors in tleap execution")
                    else:
                        print(line)
            self.reset()

        def reset(self):
            self.command = ""

        @staticmethod
        def get_n_waters(name, folder=''):
            leap = open(folder + "leap.log", 'r')
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
            return n_waters

    @staticmethod
    def pdb4amber(input, output, keep_water=False, remove_hydrogen=True, add_missing_atoms=True, reduce=False):
        comm = "pdb4amber -i " + input + " -o " + output + " "
        if remove_hydrogen:
            comm += " --nohyd "
        if not keep_water:
            comm += " --dry "
        if add_missing_atoms:
            comm += " --add-missing-atoms "
        if reduce:
            comm += " --reduce on "

        comm += "\n"
        out = os.popen(comm).read()

    @staticmethod
    def amb2gro_top_gro(file_prmtop, file_prmcrd, out_top, out_gro, out_pdb):
        os.system('amb2gro_top_gro.py -p ' + file_prmtop + '.prmtop -c ' + file_prmcrd + '.prmcrd -t ' + out_top + \
                  '.top -g ' + out_gro + '.gro -b ' + out_pdb + '.pdb')

    @staticmethod
    def antechamber(input_pdb, input_mol2, lig_name, nc=0, grms_tol=0.01, scfconv="1.d-6", ndiis_attempts=700,
                    atom_type="gaff"):
        out = os.popen(
            'antechamber -fi pdb -fo mol2 -i ' + input_pdb + '.pdb -o ' + input_mol2 + '.mol2 -at ' + atom_type +
            ' -c bcc -nc ' + str(nc) + ' -rn ' + lig_name).read()

        # ' -ek "qm_theory=\'AM1\', grms_tol=' + str(grms_tol) + ', scfconv=' + str(scfconv) + ', ndiis_attempts=' + str(ndiis_attempts) + ',"'
        # print(out)
        return out

    @staticmethod
    def parmchk2(input_mol2, out_frcmod):
        out = os.popen('parmchk2 -i ' + input_mol2 + '.mol2 -o ' + out_frcmod + '.frcmod -f mol2').read()


class FileUtility:
    @staticmethod
    def mol2Topdb(pdb_file, mol2_file, working_dir="./"):
        with open(os.path.join(working_dir, "chimera.com"), "w") as com:
            com.write("open {}\n".format(mol2_file))
            com.write("write format pdb #0 {}\n".format(pdb_file))
            com.write("stop\n")
        os.system("chimera --nogui << read {} ".format(os.path.join(working_dir, "chimera.com")))

    @staticmethod
    def get_charges_mol2(mol2_file):
        charges = []
        with open(mol2_file, "r") as file:
            while True:
                line = file.readline()
                if line.startswith("@<TRIPOS>ATOM"):
                    line = file.readline()
                    while not line.startswith("@<TRIPOS>BOND"):
                        words = line.split()
                        ch = float(words[len(words) - 1])
                        charges.append(ch)
                        line = file.readline()
                        if file.tell() == os.fstat(file.fileno()).st_size:
                            break
                if file.tell() == os.fstat(file.fileno()).st_size:
                    break
        return charges

    @staticmethod
    def rotate_files(pdb_file, mol2_file=None, new_pdb_file=None, new_mol2_file=None, translation=np.zeros(3),
                     rotation=None):
        if rotation is None:
            rotation = FileUtility.get_random_rotation()
        pdb = parsePDB(pdb_file)
        transf = Transformation(rotation, translation)
        transf.apply(pdb)
        if new_pdb_file is None:
            writePDB(pdb_file, pdb)
        else:
            writePDB(new_pdb_file, pdb)

        if mol2_file is not None:
            FileUtility.set_coords_mol2(pdb.getCoords(), mol2_file, new_mol2_file)
        return rotation, translation

    @staticmethod
    def set_coords_mol2(coords_new, file_mol2, new_mol2=None):
        lines = []
        n = 0
        with open(file_mol2, "r") as file:
            while True:
                line = file.readline()
                lines.append(line)
                if line.startswith("@<TRIPOS>ATOM"):
                    line = file.readline()
                    while not line.startswith("@<TRIPOS>BOND"):
                        #      1 C1          -7.5060   -19.3570   -23.2410 c          1 RX4       0.663100
                        new = list(line)
                        new[18:49] = list("{: 9.4f}  {: 9.4f}  {: 9.4f}".format(coords_new[n][0],
                                                                                coords_new[n][1],
                                                                                coords_new[n][2]))
                        new_line = "".join(new)
                        lines.append(new_line)
                        n += 1
                        line = file.readline()
                        if file.tell() == os.fstat(file.fileno()).st_size:
                            break
                    lines.append(line)
                if file.tell() == os.fstat(file.fileno()).st_size:
                    break
        if new_mol2 is not None:
            out = open(new_mol2, "w")
        else:
            out = open(file_mol2, "w")
        out.writelines(lines)
        out.close()

    @staticmethod
    def get_random_rotation():
        r, theta, phi = 1.0, (np.random.random() * 2.0 - 1.0) * np.pi, np.random.random() * 2 * np.pi

        x = r * cos(theta) * sin(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(phi)

        return R.from_rotvec(np.random.random() * 2 * np.pi * np.array([x, y, z])).as_matrix()

    @staticmethod
    def report(file, *args):
        out = open(file, "a")
        st = ""
        for a in args:
            st += str(a) + " "

        st += "\n"
        out.write(st)

    @staticmethod
    def remove_line(pdb_prefix, first_word, index=0):
        pdb = open(pdb_prefix + '.pdb', 'r')
        pdb_fix = open(pdb_prefix + '_temp.pdb', 'w')
        for line in pdb:
            words = line.split()
            if len(words) > index:
                if words[index] != first_word:
                    pdb_fix.write(line)
        pdb.close()
        pdb_fix.close()
        os.rename(pdb_prefix + "_temp.pdb", pdb_prefix + ".pdb")

    @staticmethod
    def rename_res(filename, from_name, to_name):
        os.system("sed -i \'s/" + from_name + "/" + to_name + "/g\' " + filename)

    @staticmethod
    def read_log(log_file, line='1', col=1):
        with open(log_file, 'r') as log:
            for lines in log:
                words = lines.split()
                if len(words) >= 3:
                    if words[0] == line:
                        words = np.array([float(words[i]) for i in range(len(words))])
                        return words[col]

    @staticmethod
    def enumerate_atom(file, atom_name):
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

    @staticmethod
    def prepare_amber_complex(cmplx_prefix, clean_prefix, lig_prefix, ligand_name):

        # rename_res(clean_prefix + '.pdb', 'CL', 'Cl')
        # rename_res(clean_prefix + '.pdb', 'FAX', 'F  ')
        # rename_res(clean_prefix + '.pdb', 'FAY', 'F  ')
        # rename_res(clean_prefix + '.pdb', 'FCX', 'F  ')
        # rename_res(clean_prefix + '.pdb', 'BR', 'Br')
        # rename_res(protein_prefix + '.pdb', 'ZN', 'Zn')
        # rename_res(clean_prefix + '.pdb', 'FE2', 'FE ')
        # rename_res(protein_prefix + '.pdb', 'CA', 'Ca')
        alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q']
        pdb = open(clean_prefix + '.pdb', 'r')
        pdb_lig = open(lig_prefix + '.pdb', 'r')
        pdb_fix = open(cmplx_prefix + '_temp.pdb', 'w')

        residues = (
            'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN',
            'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'HYS')  # 'CL', 'ZN', 'BR', 'CA', 'FE')
        chain = 0
        found = False
        write_ter = True
        for line in pdb:
            words = line.split()
            remove = False
            if len(line) > 21:
                t = list(line)
                t[21] = alpha[chain]
                line = ''.join(t)

            if words[0] != 'END':
                pdb_fix.write(line)

            adj_chain = False
            if len(words) > 3:
                if words[0] == 'TER':  # or (words[0] == 'CONECT'):
                    write_ter = False
                    adj_chain = True
                    chain += 1
                if words[2] == 'OXT' and (not adj_chain):
                    chain += 1
        if write_ter:
            pdb_fix.write('TER\n')
        chain += 1

        for line in pdb_lig:
            if len(line) > 21:
                t = list(line)
                # t[21] = alpha[chain] TODO sometimes out of range
                line = ''.join(t)
            words = line.split()
            if words[0] != 'REMARK':
                pdb_fix.write(line)

        pdb_fix.write('END\n')

        pdb_fix.close()
        pdb.close()
        pdb_lig.close()
        os.rename(cmplx_prefix + "_temp.pdb", cmplx_prefix + ".pdb")

    @staticmethod
    def get_geometric_center(selected: Selection):
        pos = selected.getCoords()
        return [np.mean(pos[:, i]) for i in range(3)]

    @staticmethod
    def top_to_itp(top_name, itp_name):
        itp_field = (
            "moleculetype", "bondtypes", "angletypes", "dihedraltypes", "atomtypes", "atoms", "bonds", "angles",
            "dihedrals", "pairs")
        not_field = ("#include", "#ifdef", "#endif")
        top = open(top_name, 'r')
        itp = open(itp_name, 'w')
        actual_field = ''
        for line in top:
            words = line.split()
            if '[' in words:
                actual_field = words[1]
            if len(words) == 2 and (words[0] in not_field):
                actual_field = ''
            if actual_field in itp_field:
                itp.write(line)
        itp.close()

        def set_box(pdb, factor=1.0):
            nanometer = nano * meter
            maxSize_x = (max((pos[0] for pos in pdb.positions)) - min((pos[0] for pos in pdb.positions))).value_in_unit(
                nanometer)
            maxSize_y = (max((pos[1] for pos in pdb.positions)) - min((pos[1] for pos in pdb.positions))).value_in_unit(
                nanometer)
            maxSize_z = (max((pos[2] for pos in pdb.positions)) - min((pos[2] for pos in pdb.positions))).value_in_unit(
                nanometer)

            vectors = (Vec3(maxSize_x * factor, 0, 0), Vec3(0, maxSize_y * factor, 0), Vec3(0, 0, maxSize_z * factor))
            pdb.topology.setPeriodicBoxVectors(vectors * nanometer)


ChemicalReaction = namedlist('ChemicalReaction', 'parent_smiles parent_id reaction zinc_id')
LigandInfo = namedlist('LigandInfo', 'smiles chemical_reaction score compound_id run_id')


class Ligand:

    def __init__(self, ligand, ligand_name, working_dir="./", id=None, generate_mol2=False):
        self.generation_info = [None, -1, None]
        self.name = ligand_name
        self.working_dir = working_dir
        self.smile = None
        self.id = id

        self.sdf_file = os.path.join(self.working_dir, self.name + ".sdf")
        self.prmtop_file = os.path.join(self.working_dir, self.name + ".prmtop")
        self.prmcrd_file = os.path.join(self.working_dir, self.name + ".prmcrd")
        self.fcrmod_file = os.path.join(self.working_dir, self.name + ".fcrmod")
        self.mol2_file = os.path.join(self.working_dir, self.name + ".mol2")
        self.prefix = os.path.join(self.working_dir, self.name)
        self.pdb_file = os.path.join(self.working_dir, self.name + ".pdb")
        self.mol_ob = None

        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        if isinstance(ligand, str):
            if ".pdb" in ligand:
                os.system("cp {} {}".format(ligand, self.pdb_file))
                Ligand.get_asymmetric_unit(self.pdb_file)
                mol = AllChem.MolFromPDBFile(ligand)
                mol_h = Chem.AddHs(mol, addCoords=True)
                # AllChem.ConstrainedEmbed(mol_h, mol)
                Chem.MolToPDBFile(mol_h, self.pdb_file)
                self.Mol = mol_h
            elif ".mol2" in ligand:
                self.mol2Topdb(self.pdb_file, ligand)
                self.Mol = AllChem.MolFromPDBFile(self.pdb_file)

            else:
                self.Mol = Chem.MolFromSmiles(ligand)
                obConversion = ob.OBConversion()
                obConversion.SetInAndOutFormats("smi", "mdl")
                self.mol_ob = ob.OBMol()
                obConversion.ReadString(self.mol_ob, ligand)
                self.mol_ob.AddHydrogens()
                self.smile = ligand
                if self.Mol is None:
                    self.Mol = Chem.MolFromInchi(ligand)
                self.Mol = Chem.AddHs(self.Mol)
                AllChem.EmbedMolecule(self.Mol)
        elif isinstance(ligand, AllChem.Mol):
            self.Mol = ligand
            self.Mol = Chem.AddHs(self.Mol)
            AllChem.EmbedMolecule(self.Mol)

            Chem.MolToPDBFile(self.Mol, self.pdb_file)
        else:
            print("error can not read and initialize the ligand")

        if self.smile is None and self.Mol is not None:
            self.smile = AllChem.MolToSmiles(Chem.RemoveHs(self.Mol))

        with Chem.SDWriter(self.sdf_file) as w:
            w.write(self.Mol)

        Chem.MolToPDBFile(self.Mol, self.pdb_file)
        # self.sanitize_pdb()

        self.prefix_solvate = None
        self.gromacs_prefix = None

        leap = AmberUtility.Leap(self.working_dir)
        if generate_mol2:
            if not os.path.exists(self.mol2_file):
                ch = Chem.GetFormalCharge(self.Mol)
                out = AmberUtility.antechamber(self.prefix, self.prefix, self.name, nc=ch, ndiis_attempts=700)
                if not os.path.exists(self.mol2_file):
                    print(out)
                    raise Exception("antechamber fail")

            AmberUtility.parmchk2(self.prefix, self.prefix)

            leap.load_gaff()
            leap.load_ff99SBildn()
            leap.load_amberparams(self.prefix)
            leap.load_mol2(ligand_name, self.prefix)
            leap.saveamberparm(ligand_name, self.prefix)
            leap.solvate(ligand_name)
            leap.execute()

    def sanitize_pdb(self):
        FileUtility.rename_res(self.pdb_file, 'UNL', self.name)
        os.system("cp {} {} ".format(self.pdb_file, self.prefix + "_temp.pdb"))
        out = open(self.prefix + ".pdb", "w")
        elements = {}
        with open(self.prefix + "_temp.pdb", "r") as file:
            for line in file:
                if line.startswith("HETATM") or line.startswith("ATOM"):
                    new = list(line)
                    element = ''.join(new[77:len(new) - 1])
                    if element in elements.keys():
                        elements[element] += 1
                    else:
                        elements[element] = 1
                    new[12:17] = "{:^5s}".format(element + str(elements[element]))
                    new[25] = '1'
                    new[21] = 'A'
                    new_line = ''.join(new)
                    out.write(new_line)
                elif line.startswith("TER"):
                    new = list(line)
                    new[25] = '1'
                    new[21] = 'A'
                    new_line = ''.join(new)
                    out.write(new_line)
                else:
                    out.write(line)
        out.close()

        fixer = PDBFixer(filename=self.prefix + ".pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(self.prefix + ".pdb", 'w'), keepIds=False)
        os.system("cp {} {} ".format(self.pdb_file, self.prefix + "_temp.pdb"))
        AmberUtility.pdb4amber(self.prefix + "_temp.pdb", self.pdb_file, remove_hydrogen=False, add_missing_atoms=False)

        os.system("rm {}".format(self.prefix + "_*"))
        # FileUtility.rename_res(self.prefix + '.pdb', ' CD ', ' Cd ')
        FileUtility.rename_res(self.prefix + '.pdb', 'CL', 'Cl')
        FileUtility.rename_res(self.prefix + '.pdb', 'BR', 'Br')

    def __str__(self):
        return AllChem.MolToSmiles(AllChem.RemoveHs(self.Mol))

    @staticmethod
    def get_asymmetric_unit(pdb_name):
        pdb = parsePDB(pdb_name)
        chains = []
        atoms_to_remove = []
        atoms_name = []
        for chain in pdb.iterChains():
            chains.append(chain.getChid())

        selection_str = "chindex 0"
        writePDB(pdb_name, pdb.select(selection_str))
        prefix = pdb_name.split(".")[0]
        os.system("cp {} {} ".format(pdb_name, prefix + "_temp.pdb"))
        out = open(pdb_name, "w")
        n = 1
        with open(prefix + "_temp.pdb", "r") as file:
            for line in file:
                if line.startswith("HETATM"):
                    new = list(line)
                    atom_name = line[13:16]
                    if atom_name not in atoms_name:
                        atoms_name.append(atom_name)
                        new[7:11] = list("{:4d}".format(n))
                        new[25] = '1'
                        new[21] = 'A'
                        new_line = ''.join(new)
                        out.write(new_line)
                        n += 1
                elif line.startswith("TER"):
                    new = list(line)
                    new[25] = '1'
                    new[21] = 'A'
                    new_line = ''.join(new)
                    out.write(new_line)
                else:
                    out.write(line)
        out.close()
        os.system("rm {}".format(prefix + "_*"))

    @staticmethod
    def get_prefix(ligand_name, working_dir):
        return os.path.join(working_dir + "/" + ligand_name)

    def solvate(self):

        self.prefix_solvate = self.working_dir + "/" + self.name + "/solvate/" + self.name

        if os.path.exists(self.working_dir + "/" + self.name + "/solvate"):
            os.system("rm -r " + self.working_dir + "/" + self.name + "/solvate")
        os.mkdir(self.working_dir + "/" + self.name + "/solvate")

        n_waters_clean = AmberUtility.Leap.get_n_waters("clean")

        print("N waters molecules added", n_waters_clean)

        leap = AmberUtility.Leap(self.working_dir)

        leap.load_gaff()
        leap.load_ff99SBildn()
        leap.load_water_spce()

        ions_water = 0.0027

        leap.load_pdb("ligand", self.prefix)
        leap.solvate("ligand",
                     n_ions=(floor(n_waters_clean * ions_water * 0.8), floor(n_waters_clean * ions_water * 0.8)))
        leap.addIons("ligand", "K+", 0)
        leap.addIons("ligand", "Cl-", 0)
        leap.saveamberparm("ligand", self.prefix_solvate + "_w")
        leap.execute()

    def generate_gromacs_files(self):
        self.gromacs_prefix = self.working_dir + "/" + self.name + "/gromacs/" + self.name
        AmberUtility.amb2gro_top_gro(self.prefix, self.prefix, self.gromacs_prefix, self.gromacs_prefix,
                                     self.gromacs_prefix)

    def draw(self, output_file=None, legend=None):
        if output_file is None:
            output_file = os.path.join(self.working_dir, self.name + "_2d.png")
        if legend is None:
            legend = self.name
        img = Draw.MolsToGridImage([self.Mol], legends=[legend])
        img.save(output_file)

    def mol2Topdb(self, pdb_file, mol2_file):
        with open(os.path.join(self.working_dir, "chimera.com"), "w") as com:
            com.write("open {}\n".format(mol2_file))
            com.write("write format pdb #0 {}\n".format(pdb_file))
            com.write("stop\n")
        os.system("chimera --nogui << read {} ".format(os.path.join(self.working_dir, "chimera.com")))

    def set_info(self, parent_1, parent_2, reaction):
        """
        Set info with no gender assumption
        :param parent_1:
        :param parent_2:
        :param reaction:
        :return:
        """
        self.generation_info = [parent_1, reaction, parent_2]
        with open(self.prefix + "_history.txt", "w") as f:
            f.write("parent_1 smile: " + str(self.generation_info[0].smile) + "\n" +
                    "parent_1 name : " + str(self.generation_info[0].id) + "\n" +
                    "reaction: " + str(self.generation_info[1]) + "\n" +
                    "parent_2 smile: " + str(self.generation_info[2].smile) + "\n" +
                    "parent_2 name : " + str(self.generation_info[2].id) + "\n")
            f.write("smile: " + str(self.smile) + "\n")

    def get_info(self):
        return self.generation_info, self.prefix + "_history.txt"


class Protein:
    class Protein_info:
        def __init__(self, protein_name, working_dir="./"):
            self.name = protein_name
            self.working_dir = working_dir
            self.prefix = os.path.join(self.working_dir, self.name)
            self.prmtop_file = self.prefix + ".prmtop"
            self.prmcrd_file = self.prefix + ".prmcrd"
            self.fcrmod_file = self.prefix + ".fcrmod"
            self.pdb_file = self.prefix + ".pdb"

    def __init__(self, protein, protein_name, keep_ions=True, working_dir="./", max_chains=1, chains_id=None):
        self.name = protein_name
        self.working_dir = working_dir
        self.prefix = os.path.join(self.working_dir, self.name)
        self.pdb_file = self.prefix + ".pdb"

        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        if not (".pdb" in protein):
            pdb_name = protein + ".pdb"
        else:
            pdb_name = protein

        os.system('rm leap.log')
        pdb = parsePDB(protein.lower())

        if keep_ions:
            writePDB(self.pdb_file, pdb.select('protein or ion'))
        else:
            writePDB(self.pdb_file, pdb.select('protein'))
        if (max_chains > 0) or (chains_id is not None):
            Protein.get_asymmetric_unit(self.pdb_file, max_chains=max_chains, chains_id=chains_id)
        AmberUtility.pdb4amber(self.pdb_file, self.prefix + "_temp.pdb", keep_water=False, remove_hydrogen=True,
                               add_missing_atoms=True)
        os.system("cp " + self.prefix + "_temp.pdb " + self.prefix + ".pdb")
        FileUtility.remove_line(self.prefix, "CONECT", 0)

        self.prmtop_file = self.prefix + ".prmtop"
        self.prmcrd_file = self.prefix + ".prmcrd"
        self.fcrmod_file = self.prefix + ".fcrmod"
        self.mol2_file = self.prefix + ".mol2"
        self.prefix_solvate = None
        self.gromacs_prefix = None

        leap = AmberUtility.Leap(working_dir)

        os.system("rm {}_*".format(self.prefix))

        leap.load_gaff()
        leap.load_ff99SBildn()
        if keep_ions:
            leap.load_water_spce()
        leap.load_pdb("protein", self.prefix)
        leap.saveamberparm("protein", self.prefix)
        leap.solvate("protein")
        leap.execute()

    def solvate(self):

        ions_water = 0.0027
        self.prefix_solvate = self.working_dir + "/" + self.name + "/solvate/" + self.name

        if os.path.exists(self.working_dir + "/" + self.name + "/solvate"):
            os.system("rm -r " + self.working_dir + "/" + self.name + "/solvate")
        os.mkdir(self.working_dir + "/" + self.name + "/solvate")

        leap = AmberUtility.Leap(self.working_dir)

        n_waters_clean = AmberUtility.Leap.get_n_waters("clean")

        print("N Waters molecules added", n_waters_clean)

        leap.load_gaff()
        leap.load_ff99SBildn()
        leap.load_water_spce()
        leap.load_pdb("protein", self.prefix_solvate)
        leap.solvate("protein",
                     n_ions=(floor(n_waters_clean * ions_water * 0.8), floor(n_waters_clean * ions_water * 0.8)))
        leap.addIons("protein", "K+", 0)
        leap.addIons("protein", "Cl-", 0)
        leap.saveamberparm("protein", self.prefix_solvate + "_w")
        leap.execute()

    def generate_gromacs_files(self):
        self.gromacs_prefix = self.working_dir + "/" + self.name + "/gromacs/" + self.name
        AmberUtility.amb2gro_top_gro(self.prefix, self.prefix, self.gromacs_prefix, self.gromacs_prefix,
                                     self.gromacs_prefix)

    @staticmethod
    def get_asymmetric_unit(pdb_name, max_chains=1, chains_id=None):
        threshold = 98
        pdb = parsePDB(pdb_name)
        chains = []
        chains_to_remove = []
        for chain in pdb.iterChains():
            chains.append([chain.getSequence(), chain.getChid()])

        if chains_id is not None:
            selection_str = "chindex"
            for i in range(len(chains_id)):
                selection_str += " " + str(chains_id[i])
            writePDB(pdb_name, pdb.select(selection_str))
            return

        if len(chains) > max_chains:
            for i in range(len(chains)):
                for j in range(i + 1, len(chains)):
                    if (i not in chains_to_remove) and (j not in chains_to_remove):
                        match = Protein.sequences_match(chains[i][0], chains[j][0])
                        print(match)
                        if match >= threshold:
                            chains_to_remove.append(j)

            selection_str = "chindex"
            for i in range(len(chains)):
                if i not in chains_to_remove:
                    selection_str += " " + str(i)

            print(selection_str)
            writePDB(pdb_name, pdb.select(selection_str))

    @staticmethod
    def sequences_match(first_seq, second_seq):
        global_align = pw2.align.globalxx(first_seq, second_seq)
        seq_length = min(len(first_seq), len(second_seq))
        matches = global_align[0][2]
        percent_match = (matches / seq_length) * 100
        return percent_match


class Crystal:
    class Crystal_info:
        def __init__(self, protein_name, ligand_name, ligand, working_dir="./"):
            self.working_dir = working_dir
            if not os.path.exists(os.path.join(self.working_dir, "ligand")):
                os.mkdir(os.path.join(self.working_dir, "ligand"))
            if not os.path.exists(os.path.join(self.working_dir, "protein")):
                os.mkdir(os.path.join(self.working_dir, "protein"))
            if not os.path.exists(os.path.join(self.working_dir, "complex")):
                os.mkdir(os.path.join(self.working_dir, "complex"))

            self.name = protein_name + "_complex_" + ligand_name
            self.prefix = os.path.join(self.working_dir, "complex", protein_name + "_complex_" + ligand_name)
            self.protein = Protein.Protein_info(protein_name, working_dir=os.path.join(self.working_dir, "protein"))
            self.ligand = Ligand.Ligand_info(ligand, ligand_name, working_dir=os.path.join(self.working_dir, "ligand"))
            self.pdb_file = self.prefix + ".pdb"

    def __init__(self, working_dir="./"):
        self.working_dir = working_dir
        self.prefix = None
        self.pdb_file = None
        self.protein = None
        self.ligand = None
        self.name = None
        if not os.path.exists(os.path.join(self.working_dir)):
            os.mkdir(os.path.join(self.working_dir))

    def generate_pqr(self):
        apbs = APBSUtility(self.working_dir)
        apbs.mol22pqr(self.ligand.mol2_file, None, self.ligand.prefix + ".pqr", self.ligand.name, self.ligand.pdb_file)
        apbs.pdb2pqr(self.protein.pdb_file, self.protein.prefix + ".pqr")
        apbs.complex2pqr(self.protein.prefix + ".pqr", self.prefix + ".pqr", self.ligand.prefix + ".pqr", self.ligand.name)

    def load_protein_ligand(self, protein_pdb, ligand_mol2, protein_name, ligand_name):
        if not os.path.exists(os.path.join(self.working_dir, "ligand")):
            os.mkdir(os.path.join(self.working_dir, "ligand"))
        if not os.path.exists(os.path.join(self.working_dir, "protein")):
            os.mkdir(os.path.join(self.working_dir, "protein"))
        if not os.path.exists(os.path.join(self.working_dir, "complex")):
            os.mkdir(os.path.join(self.working_dir, "complex"))
        self.prefix = os.path.join(self.working_dir, "complex", protein_name + "_complex_" + ligand_name)
        self.pdb_file = self.prefix + ".pdb"
        self.name = protein_name + "_complex_" + ligand_name
        pdb_pro = parsePDB(protein_pdb)
        writePDB(os.path.join(self.working_dir, "protein", protein_name + ".pdb"),
                 pdb_pro.select("protein or ion"))

        self.ligand = Ligand(ligand_mol2, ligand_name, working_dir=os.path.join(self.working_dir, "ligand"))
        self.protein = Protein(os.path.join(self.working_dir, "protein", protein_name + ".pdb"), protein_name,
                               working_dir=os.path.join(self.working_dir, "protein"),
                               max_chains=-1, chains_id=self.get_chains_in_binding_pocket(protein_pdb, self.ligand.pdb_file))

        FileUtility.prepare_amber_complex(self.prefix + '_temp', self.protein.prefix,
                                          self.ligand.prefix, self.ligand.name)

        AmberUtility.pdb4amber(self.prefix + "_temp.pdb", self.prefix + ".pdb", keep_water=False,
                               remove_hydrogen=False, add_missing_atoms=True)

        os.system('rm ' + self.prefix + '_*')
        FileUtility.remove_line(self.prefix, "CONECT", 0)
        os.system('rm leap.log')

        leap = AmberUtility.Leap()
        leap.load_gaff()
        leap.load_ff99SBildn()
        leap.load_water_spce()
        leap.load_mol2(self.ligand.name, self.ligand.prefix)
        leap.load_amberparams(self.ligand.prefix)
        leap.load_pdb("complex", self.prefix)
        leap.saveamberparm("complex", self.prefix)
        leap.execute()

    def get_chains_in_binding_pocket(self, pdb, ligand_pdb):
        ligand_coords = parsePDB(ligand_pdb).getCoords()
        center = [np.average(ligand_coords[:, i]) for i in range(3)]
        sel = parsePDB(pdb).select("protein and ( ( ( x - {} ) ** 2 + ( y - {} ) ** 2 + ( z - {} ) ** 2 ) < (10.0 ** 2 ) )".format(
            center[0], center[1], center[2]))
        id = sel.getChindices()
        res = []
        for i in id:
            if i not in res:
                res.append(i)
        return res

    def load_crystal_pdb(self, pdb_file, ligand_name, protein_name):
        crystal_pdb_pro = parsePDB(pdb_file)
        if not os.path.exists(os.path.join(self.working_dir, "ligand")):
            os.mkdir(os.path.join(self.working_dir, "ligand"))
        if not os.path.exists(os.path.join(self.working_dir, "protein")):
            os.mkdir(os.path.join(self.working_dir, "protein"))
        if not os.path.exists(os.path.join(self.working_dir, "complex")):
            os.mkdir(os.path.join(self.working_dir, "complex"))

        self.prefix = os.path.join(self.working_dir, "complex", protein_name + "_complex_" + ligand_name)
        self.pdb_file = self.prefix + ".pdb"
        writePDB(os.path.join(self.working_dir, protein_name + ".pdb"),
                 crystal_pdb_pro.select("protein or ion or ( " + "resname " + ligand_name + ")"))
        writePDB(os.path.join(self.working_dir, "protein", protein_name + ".pdb"),
                 crystal_pdb_pro.select("protein or ion"))
        writePDB(os.path.join(self.working_dir, "ligand", ligand_name + ".pdb"),
                 crystal_pdb_pro.select("resname " + ligand_name))

        self.protein = Protein(os.path.join(self.working_dir, "protein", protein_name + ".pdb"), protein_name,
                               working_dir=os.path.join(self.working_dir, "protein"),
                               max_chains=Crystal.get_max_chains(os.path.join(self.working_dir, protein_name + ".pdb"),
                                                                 ligand_name))

        self.ligand = Ligand(os.path.join(self.working_dir, "ligand", ligand_name + ".pdb"), ligand_name,
                             working_dir=os.path.join(self.working_dir, "ligand"))

        FileUtility.prepare_amber_complex(self.prefix + '_temp', self.protein.prefix,
                                          self.ligand.prefix, self.ligand.name)

        AmberUtility.pdb4amber(self.prefix + "_temp.pdb", self.prefix + ".pdb", keep_water=False,
                               remove_hydrogen=False, add_missing_atoms=True)

        os.system('rm ' + self.prefix + '_*')
        FileUtility.remove_line(self.prefix, "CONECT", 0)
        os.system('rm leap.log')

        leap = AmberUtility.Leap()
        leap.load_gaff()
        leap.load_ff99SBildn()
        leap.load_water_spce()
        leap.load_mol2(self.ligand.name, self.ligand.prefix)
        leap.load_amberparams(self.ligand.prefix)
        leap.load_pdb("complex", self.prefix)
        leap.saveamberparm("complex", self.prefix)
        leap.execute()

    @staticmethod
    def get_max_chains(pdb_file, ligand_name):
        atoms_name = []
        n_ligands = 0
        check = None
        with open(pdb_file, "r") as file:
            for line in file:
                if line.startswith("HETATM"):
                    new = list(line)
                    atom_name = line[13:16]
                    res = line[17:20]
                    if res == ligand_name:
                        if check is None:
                            check = atom_name
                        if check == atom_name:
                            n_ligands += 1
        pdb = parsePDB(pdb_file)
        print(n_ligands, pdb.numChains())
        return int(pdb.numChains() / n_ligands)


class Complex:

    def __init__(self, protein: Protein, ligand: Ligand, complex_name, working_dir="./", domain_selection=None, use_mol2=False, generate_files=False):

        self.ligand = ligand
        self.protein = protein
        self.path_mlg = os.environ.get("PATH_MLG")
        if self.path_mlg is None:
            self.path_mlg = '/home/' + os.environ.get(
                "USER") + '/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'

        self.name = complex_name
        self.working_dir = working_dir
        self.prefix = self.working_dir + "/" + self.name
        self.prefix_protein = self.working_dir + "/" + self.name + "_protein"
        self.prefix_ligand = self.working_dir + "/" + self.name + "_ligand"

        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        Chem.MolToPDBFile(ligand.Mol, self.prefix_ligand + ".pdb")
        FileUtility.rename_res(self.prefix_ligand + '.pdb', 'UNL', 'LIG')

        if domain_selection is None:
            positions = parsePDB(protein.pdb_file).getCoords()
        else:
            print(domain_selection)
            positions = parsePDB(protein.pdb_file).select(domain_selection).getCoords()
        positions = np.concatenate((positions, parsePDB(ligand.pdb_file).getCoords()))

        # compute the box size for autodock vina
        box_size = [(max((pos[i] for pos in positions)) - min((pos[i] for pos in positions))) * 2 for i in range(3)]
        box_center = [np.average([pos[i] for pos in positions]) for i in range(3)]
        num_modes = 3
        whole_selection = 'protein or (resname ' + ligand.name + ')'
        os.system('rm leap.log')

        AmberUtility.pdb4amber(protein.pdb_file, self.prefix_protein + ".pdb", keep_water=False,
                               remove_hydrogen=True, add_missing_atoms=True)

        # create n_modes docking configuration via autodock
        with open('script.com', 'w') as script:
            script.write('ligand = ' + self.prefix_ligand + '.pdbqt\n' +
                         'receptor = ' + self.prefix_protein + '.pdbqt\n' +
                         'center_x = ' + str(box_center[0]) + '\n' +
                         'center_y = ' + str(box_center[1]) + '\n' +
                         'center_z = ' + str(box_center[2]) + '\n' +
                         'size_x = ' + str(box_size[0]) + '\n' +
                         'size_y = ' + str(box_size[1]) + '\n' +
                         'size_z = ' + str(box_size[2]) + '\n' +
                         'out = ' + self.prefix_ligand + '_vina.pdbqt\n' +
                         'num_modes = ' + str(num_modes) + '\n' +
                         'exhaustiveness = 32\n' +
                         'log = ' + self.working_dir + '/log.txt\n')

        with open('script.sh', 'w') as script:
            if use_mol2:
                script.write('#!/bin/bash\n' +
                             'prepare_ligand -l ' + ligand.prefix + '.mol2 -A \'bonds\' -U \'lps\' -o ' + self.prefix_ligand + '.pdbqt -C \n' +
                             'prepare_receptor -r ' + self.prefix_protein + '.pdb -o ' + self.prefix_protein + '.pdbqt\n')
            else:
                script.write('#!/bin/bash\n' +
                             'mk_prepare_ligand.py -i ' + ligand.prefix + '.sdf -o ' + self.prefix_ligand + '.pdbqt \n' +
                             'prepare_receptor -r ' + self.prefix_protein + '.pdb -o ' + self.prefix_protein + '.pdbqt\n')

        os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/ADFRsuite-1.0/bin'
        os.system('bash script.sh')
        FileUtility.rename_res(self.prefix_ligand + '.pdbqt', 'Ho', 'HD')

        v = Vina(sf_name='vina', cpu=cpu_count())
        v.set_receptor(f'{self.prefix_protein}.pdbqt')
        v.set_ligand_from_file(f'{self.prefix_ligand}.pdbqt')
        v.compute_vina_maps(center=box_center, box_size=box_size)
        v.optimize()
        v.dock(exhaustiveness=8, n_poses=num_modes)
        v.write_poses(f'{self.prefix_ligand}_vina.pdbqt', n_poses=num_modes, overwrite=True)

        with open('script.sh', 'w') as script:
            script.write('#!/bin/bash\n' +
                         'vina_split --input ' + self.prefix_ligand + '_vina.pdbqt --ligand ' + self.prefix_ligand + '_ \n')
        os.system('bash script.sh')
        energies = v.energies()

        modes = []
        dG_0 = np.min(energies[:, 0])
        for i in range(1, num_modes + 1):
            if os.path.exists(self.prefix_ligand + '_' + str(i) + '.pdbqt'):
                # elif (np.fabs(dG - dG_0) < 0.15) and (rmsd_l >= 3.0) and (rmsd_l < 50):

                dG = energies[i - 1]
                os.system(f"babel -ipdbqt '{self.prefix_ligand}'_'{str(i)}.pdbqt' -opdb '{self.prefix_ligand}'_'{str(i)}.pdb'")
                FileUtility.enumerate_atom(self.prefix_ligand + '_' + str(i), 'Cl')
                FileUtility.enumerate_atom(self.prefix_ligand + '_' + str(i), 'Br')
                modes.append(i)
        self.modes = modes
        self.dG_autodock = dG_0

        # modes = [1]
        if generate_files:
            # create the complex system for the receptor and the ligand
            for i in modes:
                protein_prefix = self.prefix_protein + '_' + str(i)
                ligand_prefix = self.prefix_ligand + '_' + str(i)

                AmberUtility.pdb4amber(self.prefix_protein + ".pdb", protein_prefix + "_clean.pdb", keep_water=False,
                                       remove_hydrogen=False, add_missing_atoms=True)
                fixer = PDBFixer(filename=ligand_prefix + ".pdb")
                PDBFile.writeFile(fixer.topology, fixer.positions, open(ligand_prefix + ".pdb", 'w'), keepIds=True)

                FileUtility.prepare_amber_complex(protein_prefix + '_temp', protein_prefix + "_clean",
                                                  ligand_prefix, ligand.name)

                AmberUtility.pdb4amber(protein_prefix + "_temp.pdb", protein_prefix + ".pdb", keep_water=False,
                                       remove_hydrogen=False, add_missing_atoms=True)

                os.system('rm ' + protein_prefix + '_*')
                FileUtility.remove_line(protein_prefix, "CONECT", 0)
                os.system('rm leap.log')
                pdb = parsePDB(protein_prefix + '.pdb')
                lig_center = FileUtility.get_geometric_center(pdb.select('(resname ' + ligand.name + ')'))

                leap = AmberUtility.Leap()
                leap.load_gaff()
                leap.load_ff99SBildn()
                leap.load_water_spce()
                leap.load_mol2(ligand.name, ligand.prefix)
                leap.load_amberparams(ligand.prefix)
                leap.load_pdb("complex", protein_prefix)
                leap.saveamberparm("complex", protein_prefix)
                leap.execute()


    def solvate(self):

        self.prefix_solvate = self.working_dir + "/" + self.name + "/solvate/" + self.name

        ions_water = 0.0027
        if os.path.exists(self.working_dir + "/" + self.name + "/solvate"):
            os.system("rm -r " + self.working_dir + "/" + self.name + "/solvate")
        os.mkdir(self.working_dir + "/" + self.name + "/solvate")

        n_waters_complex = AmberUtility.Leap.get_n_waters("complex")

        leap = AmberUtility.Leap()
        leap.load_gaff()
        leap.load_ff99SBildn()
        leap.load_water_spce()
        leap.load_mol2(self.ligand.name, self.ligand.prefix)
        leap.load_amberparams(self.ligand.prefix)
        leap.load_pdb("complex", self.prefix_protein)
        leap.solvate("protein",
                     n_ions=(floor(n_waters_complex * ions_water * 0.8), floor(n_waters_complex * ions_water * 0.8)))
        leap.addIons("protein", "K+", 0)
        leap.addIons("protein", "Cl-", 0)
        leap.saveamberparm("complex", self.prefix_solvate)
        leap.execute()

    def generate_gromacs_files(self, mode_n=1):
        os.system("mkdir " + os.path.join(self.working_dir, "gromacs_files"))
        AmberUtility.amb2gro_top_gro(self.protein.prefix, self.protein.prefix,
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_protein),
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_protein),
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_protein))
        AmberUtility.amb2gro_top_gro(self.ligand.prefix, self.ligand.prefix,
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_ligand),
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_ligand),
                                     os.path.join(self.working_dir, "gromacs_files", self.prefix_ligand))
        AmberUtility.amb2gro_top_gro(self.prefix_protein + "_" + str(mode_n), self.prefix_protein + "_" + str(mode_n),
                                     os.path.join(self.working_dir, "gromacs_files",
                                                  self.prefix_protein + "_" + str(mode_n)),
                                     os.path.join(self.working_dir, "gromacs_files",
                                                  self.prefix_protein + "_" + str(mode_n)),
                                     os.path.join(self.working_dir, "gromacs_files",
                                                  self.prefix_protein + "_" + str(mode_n)))

        gro = GromacsGroFile(
            os.path.join(self.working_dir, "gromacs_files", self.prefix_protein + "_" + str(mode_n) + '.gro'))
        top = GromacsTopFile(
            os.path.join(self.working_dir, "gromacs_files", self.prefix_protein + "_" + str(mode_n) + '.top'))

        bonds, angles, dihedrals, idx = self.get_costraints(top.topology, gro.getPositions(asNumpy=True),
                                                            self.ligand.name)

        with open(os.path.join(self.working_dir, "gromacs_files", self.prefix_protein + "_" + str(mode_n) + '.top',
                               'a')) as top_file:
            top_file.write("[ intermolecular_interactions ]\n" +
                           "[ bonds ]\n" +
                           "; ai     aj    type   bA      kA     bB      kB\n" +
                           str(idx[0][0]) + " " + str(idx[1][0]) + " 6 " + str(bonds[0]) + " 0.0 " + str(
                bonds[0]) + " 4184.0\n" +
                           "[ angles ]\n" +
                           "; ai     aj    ak     type    thA      fcA        thB      fcB\n" +
                           str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(idx[0][0]) + " 1 " + str(
                angles[0]) + " 0.0 " + str(angles[0]) + " 41.84\n" +
                           str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(idx[0][1]) + " 1 " + str(
                angles[1]) + " 0.0 " + str(angles[1]) + " 41.84\n" +
                           "[ dihedrals ]\n" +
                           "; ai     aj    ak    al    type     thA      fcA       thB      fcB\n" +
                           str(idx[1][2]) + " " + str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(
                idx[0][0]) + " 2 " + str(dihedrals[0]) + " 0.0 " + str(dihedrals[0]) + " 41.84\n" +
                           str(idx[1][1]) + " " + str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(
                idx[0][1]) + " 2 " + str(dihedrals[1]) + " 0.0 " + str(dihedrals[1]) + " 41.84\n" +
                           str(idx[1][0]) + " " + str(idx[0][0]) + " " + str(idx[0][1]) + " " + str(
                idx[0][2]) + " 2 " + str(dihedrals[2]) + " 0.0 " + str(dihedrals[2]) + " 41.84\n")

        os.system("gmx editconf -f ")

    @staticmethod
    def get_constraints(topology: Topology, positions, lig_name):

        def distance(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

        min_dist = 50
        couple = []
        couple_indx = []
        for residue_pro in topology.residues():
            if residue_pro.name != lig_name:
                for atom_pro in residue_pro.atoms():
                    for residue in topology.residues():
                        if residue.name == lig_name:
                            for atom_lig in residue.atoms():
                                dist = distance(positions[atom_pro.index], positions[atom_lig.index])
                                if (dist < min_dist) and (atom_pro.index != atom_lig.index) and (
                                        atom_pro.element != elem.hydrogen) and (atom_lig.element != elem.hydrogen):
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
                if len(atoms_pro) == 3:
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

        print(atoms_pro[0], atoms_pro[1], atoms_pro[2])

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

        print(atoms_lig)

        positions = positions
        dist = [calcDistance(positions[atoms_pro[0].index], positions[atoms_lig[0].index])]
        angles = [
            measure.getAngle(positions[atoms_pro[1].index], positions[atoms_pro[0].index],
                             positions[atoms_lig[0].index]),
            measure.calcAngle(positions[atoms_pro[0].index], positions[atoms_lig[0].index],
                              positions[atoms_lig[1].index])]
        dihedral = [measure.getDihedral(positions[atoms_pro[2].index], positions[atoms_pro[1].index],
                                        positions[atoms_pro[0].index], positions[atoms_lig[0].index]),
                    measure.getDihedral(positions[atoms_pro[1].index], positions[atoms_pro[0].index],
                                        positions[atoms_lig[0].index], positions[atoms_lig[1].index]),
                    measure.getDihedral(positions[atoms_pro[0].index], positions[atoms_lig[0].index],
                                        positions[atoms_lig[1].index], positions[atoms_lig[2].index])]

        return dist, angles, dihedral, [[atoms_pro[i].index for i in range(len(atoms_pro))],
                                        [atoms_lig[i].index for i in range(len(atoms_lig))]]
