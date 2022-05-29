from Samplers import *


def __generate_ligands(th_id, from_i, to_i, ligands, ligands_name, ligands_folder):
    if not os.path.exists("THREAD_" + str(th_id)):
        os.mkdir("THREAD_" + str(th_id))
    os.chdir("THREAD_" + str(th_id))
    print("running thread " + str(th_id))
    for i in range(from_i, to_i):
        print("generating ligand " + ligands_name[i] + " in " + ligands_folder[i])
        try:
            out = Ligand(ligands[i], ligands_name[i], ligands_folder[i])
        except BaseException as e:
            print(e, ligands_folder[i])
    os.chdir("../")


if __name__ == '__main__':
    n_threads = 3
    filename = '/home/giacomo/PycharmProjects/DrugDiscovery_hPARP/Lab-1.sdf'
    folder = "/home/giacomo/Documents/hPARP-1/RUN_3"
    suppl = Chem.SDMolSupplier(filename)
    molecules = []
    names = []
    id = 1
    for mol in suppl:
        try:
            names.append(mol.GetProp('MOLENAME').replace(' ', '_'))
            molecules.append(os.path.join(folder, "ligands_originals", "outputfile{}.pdb".format(id)))
            id += 1
        except:
            pass

    ligands_name = []
    ligands_folder = []
    for k in range(len(molecules)):
        ligand_folder = os.path.join(folder, "ligands", names[k])
        ligands_name.append("LIG")
        ligands_folder.append(ligand_folder)
    if not os.path.exists(os.path.join(folder, "ligands")):
        os.mkdir(os.path.join(folder, "ligands"))
    # pool = multiprocessing.Pool(processes=len(ligands)) pool.map(target=self.__generate_ligands, [(i,
    # ligands[i], ligands_name[i], ligands_folder[i], self.ligands[i],) for i in range(len(ligands))])
    # TODO Use a Pool object

    """threads = [multiprocessing.Process(target=__generate_ligands,
                                       args=(i,
                                             int(i * len(molecules) / n_threads),
                                             int((i + 1) * len(molecules) / n_threads),
                                             molecules, ligands_name, ligands_folder,))
               for i in range(n_threads)]

    for th in threads:
        th.start()

    for th in threads:
        th.join()"""

    # Evaluate the ligands
    protein = Protein('/home/giacomo/Documents/hPARP-1/hPARP-1.pdb', 'hPARP-1', True, '/home/giacomo/Documents/hPARP-1/')
    dG = []
    for k in range(0, len(molecules)):
        #if True:
        if k != 721:
            try:
                output = open('/home/giacomo/Documents/hPARP-1/out_3.dat', 'a')
                ligand = Ligand(molecules[k], "LIG", ligands_folder[k])

                complex_name = protein.name + "_" + names[k]
                complex_folder = os.path.join(folder, "complexes", complex_name)
                _complex = Complex(protein, ligand, complex_name,
                                   working_dir=complex_folder)

                output.write(str(k) + " " + ligand.smile + " " + names[k] + " " + str(_complex.dG_autodock) + "\n")
                output.close()
                dG.append(_complex.dG_autodock)
            except:
                print("error valutate autodock")



