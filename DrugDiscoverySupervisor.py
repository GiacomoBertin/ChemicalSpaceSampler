import threading as thread
from FAT import *
from autogrow.Operators.Filter.Filter_classes.FilterClasses.Ghose import *
from Reactor import *
from multiprocessing import Process, Manager
import multiprocessing as mp
import time
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import json


class DrugDiscoverySupervisor:
    def __init__(self, working_directory, protein_pdb, log_file, n_child=1000, n_threads=1, domain_selection=None,
                 n_max_parallel_process=10, lag_iterations=5, embed_dimension=3):
        self.n_child = n_child
        self.working_dir = working_directory
        self.n_threads = n_threads
        self.protein_pdb = protein_pdb
        self.ligands_info_pool = []
        self.log_file = log_file
        self.domain_selection = domain_selection
        self.fragments_directory = os.path.join(self.working_dir, "ZINC_library")
        self.n_max_parallel_process = n_max_parallel_process
        self.lag_iterations = lag_iterations
        self.embed_dimension = embed_dimension
        self.childes_smiles = []
        self.childes_info = []
        self.zinc_dictionary = {}  # ZINC_ID: Mol
        self.watchdog = None
        self.vae_sampler = None
        self.vfee_min_training_data = 0
        self.mfee_sampler = MFEESampler(working_directory=self.fragments_directory,
                                        protein_pdb=os.path.join(self.fragments_directory,
                                                                 self.protein_pdb),
                                        log_file=os.path.join(self.fragments_directory,
                                                              "log_ZINC_library_mfee.dat"),
                                        learning_rate=0.05, embed_dimension=self.embed_dimension,
                                        n_threads=self.n_threads,
                                        domain_selection=self.domain_selection)
        self.train_smiles_dg = []

    def load_train_set(self, dg, smiles):
        for i in range(len(dg)):
            if len(self.train_smiles_dg) > 0:
                my_smiles = np.array(self.train_smiles_dg)[:, 0]
            else:
                my_smiles = []
            if smiles[i] not in my_smiles:
                self.train_smiles_dg.append([smiles[i], dg[i]])

    class Watchdog:
        def __init__(self, n_max_processes):
            self.processes = [None] * n_max_processes
            self.queue = mp.Queue()
            self.n_processes_alive = 0
            self.waiting_processes = []

        def kill_job(self, i):
            print("Killing " + str(self.processes[i].name))
            self.processes[i].terminate()
            while self.processes[i].is_alive():
                time.sleep(0.1)
            self.processes[i].join(timeout=1.0)
            print("Succesfully Killed " + str(self.processes[i].name))

        def submit_process(self, target, args):
            self.waiting_processes.append(Process(target=target, args=args))
            self.check()
            print("Process succesfully queued")

        def kill_jobs(self, idx):
            for i in idx:
                self.kill_job(i)

        def join(self):
            for p in self.processes:
                p.join()

        def check_processes_alive(self):
            self.n_processes_alive = 0
            for i in range(len(self.processes)):
                if self.processes[i] is not None:
                    if not self.processes[i].is_alive():
                        self.kill_job(i)
                    else:
                        self.n_processes_alive += 1

        def get_free_position(self):
            self.check_processes_alive()
            for k in range(len(self.processes)):
                if (self.processes[k] is None) or (not self.processes[k].is_alive()):
                    return k
            return None

        def check(self):

            self.check_processes_alive()

            try:
                msg = self.queue.get_nowait()
                if msg[0] == "KILL":
                    self.kill_job(msg[1])
            except:
                print("No job to kill")

            for i in range(len(self.waiting_processes)):
                pos = self.get_free_position()
                if pos is not None:
                    if pos < len(self.processes) and i < len(self.waiting_processes):
                        self.processes[pos] = self.waiting_processes[i]
                        self.waiting_processes.remove(self.waiting_processes[i])
                        self.processes[pos].start()
                    else:
                        print("Something wrong {}, {}".format(pos, i))

    def __read_mol2_file(self, file, sanitize=True):
        mols = []
        ZINC_id = ""
        names = []
        print(file)
        with open(file, 'r') as f:
            line = f.readline()
            while not f.tell() == os.fstat(f.fileno()).st_size:
                m = None
                if line.startswith("@<TRIPOS>MOLECULE"):
                    mol = []
                    mol.append(line)
                    line = f.readline()

                    while not line.startswith("@<TRIPOS>MOLECULE"):
                        words = line.split()
                        if len(words) == 1 and "ZINC" in words[0]:
                            ZINC_id = words[0]
                        mol.append(line)
                        line = f.readline()
                        if f.tell() == os.fstat(f.fileno()).st_size:
                            mol.append(line)
                            break
                    mol[-1] = mol[-1].rstrip()  # removes blank line at file end
                    block = ",".join(mol).replace(',', '')
                    # if not os.path.exists( os.path.join(self.fragments_directory, ZINC_id)):
                    #    os.system("mkdir " + os.path.join(self.fragments_directory, ZINC_id))
                    # with open(Ligand.get_prefix(ZINC_id, os.path.join(self.fragments_directory, ZINC_id)) + ".mol2",
                    #          "w") as mol2_file:
                    #    mol2_file.writelines(mol)

                    m = Chem.MolFromMol2Block(block, sanitize=sanitize, removeHs=False)
                if ZINC_id not in names:
                    if m is not None:
                        smile = AllChem.MolToSmiles(AllChem.RemoveHs(m))
                        if len(smile) <= 120:
                            try:
                                ohf = OneHotFeaturizer()
                                ohf.featurize([smile.ljust(120)])
                                names.append(ZINC_id)
                                mols.append(m)
                            except:
                                print("Discarded: " + smile)
        return mols, names

    def __read_json_file(self, file):
        mols = []
        names = []
        with open(file, "r") as f:
            for line in f:
                words = line.split()
                if len(words) >= 2:
                    smile = words[0]
                    if len(smile) <= 120:
                        try:
                            ohf = OneHotFeaturizer()
                            ohf.featurize([smile.ljust(120)])
                            names.append(words[1])
                            mols.append(AllChem.MolFromSmiles(smile))
                        except:
                            print("Discarded: " + smile)

        return mols, names

    def load_ZINC_reagents(self, zinc_folder="./Reaction_libraries/ClickChem/complimentary_mol_dir"):
        for root, dirs, files in os.walk(zinc_folder):
            for name in files:
                file = os.path.join(root, name)
                mols, names = self.__read_json_file(file)
                keys = list(self.zinc_dictionary.keys())
                for l in range(len(mols)):
                    if names[l] not in keys:
                        self.zinc_dictionary[names[l]] = mols[l]
                pass
        print("Database loaded")

    def load_ZINC_database(self, zinc_folder):
        if not os.path.exists(os.path.join(self.fragments_directory)):
            os.system("mkdir " + os.path.join(self.fragments_directory))
        for root, dirs, files in os.walk(zinc_folder):
            for name in files:
                file = os.path.join(root, name)
                mols, names = self.__read_mol2_file(file, sanitize=False)
                for l in range(len(mols)):
                    # ligand = Ligand(mols[l], names[l], os.path.join(self.fragments_directory, names[l]))
                    self.zinc_dictionary[names[l]] = mols[l]
                pass
        print("Database loaded")

    def report(self, line):
        with open(self.log_file, "a") as file:
            for i in line:
                file.write(str(i) + " ")
            file.write("\n")

    @staticmethod
    def __cluster_fps(ms, cutoff=0.2):
        """
        Find the representative structures between the molecules passed as input. Use the Butina algorithm
        and Morgan fingerprint.

        :param ms:      Molecules to cluster
        :param cutoff:  elements within this range of each other are considered
                        to be neighbors
        :return:        a tuple of tuples containing information about the clusters:
                        ( (cluster1_elem1, cluster1_elem2, ...),
                          (cluster2_elem1, cluster2_elem2, ...),
                          ...
                        )
                        The first element for each cluster is its centroid.
        """

        # first generate the distance matrix:
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in ms]
        dists = []
        nfps = len(fps)
        print("Generate the distance matrix\n")

        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])

        # now cluster the data:
        print("Cluster the data\n")
        cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

        return np.array(cs)

    @staticmethod
    def __fingerprint_similarity(mol_1, mol_2):
        """
        returns the calculated similarity between two fingerprints,
        handles any folding that may need to be done to ensure that they are compatible

        """
        return DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_1),
                                                 Chem.RDKFingerprint(mol_2))

    def __react(self, parent_smile, n_child, i_d):

        smileClick = Reactor(["ClickChem", "", "", ""], [], {})
        childes = []

        for i in range(n_child):
            child_info = smileClick.run_Smile_Click(parent_smile)
            if child_info is None:
                continue
            else:
                childes.append(child_info)
                smileClick.update_list_of_already_made_smiles(childes)

        self.childes_smiles[i_d] = [c[0] for c in childes]
        self.childes_info[i_d] = [c for c in childes]

    @staticmethod
    def __subdivide_array(a, n):
        b = []
        count = 0
        for i in range(n):
            b.append([])
            for j in range(i * int(len(a) / n), (i + 1) * int(len(a) / n)):
                if j < len(a):
                    count += 1
                    b[i].append(a[j])

        for i in range(count, len(a)):
            b[n - 1].append(a[i])
        return b

    def validate_product(self, mol, already_computed_mol_protonated, similarity_threshold):
        filter = Ghose()
        mol_prot = AllChem.AddHs(mol)
        filter_res = filter.run_filter(mol_prot)
        ohf = OneHotFeaturizer()
        if not filter_res:
            print("Molecule does not pass Goose test")
            return False
        smiles_prot = AllChem.MolToSmiles(mol_prot)
        smile = AllChem.MolToSmiles(AllChem.RemoveHs(mol))
        try:
            ohf.featurize([smile.ljust(120)])
        except:
            print("Molecule failed featurization test")
            return False

        for s in already_computed_mol_protonated:
            d = DrugDiscoverySupervisor.__fingerprint_similarity(mol_prot, s)
            if d > (1 - similarity_threshold):
                print("Molecule too similar at another computed yet")
                return False
        return True

    def __mutation_worker(self, parent_1: Ligand, n_iterations, mfee_file, process_id,
                          ligand_dictionary, compound_number, lock, messanger):
        reactor = Reactor(["ClickChem", "", "", ""], [], {})
        print("Iter crossover for :" + str(self.n_child))
        if not os.path.exists("Process_{}".format(process_id)):
            os.mkdir("Process_{}".format(process_id))
        os.chdir("Process_{}".format(process_id))
        childes_info = []

        # generate the possible crossover products
        for i in range(self.n_child):
            child_info = reactor.run_Smile_Click(parent_1.smile)
            if child_info is None:
                continue
            else:
                childes_info.append(child_info)
                reactor.update_list_of_already_made_smiles(childes_info)
        childes_info = np.array(childes_info)
        print(len(childes_info))

        # validate the products
        # validated_products = [Mol, smiles, reaction, parent_2_name (always ZINC)]
        validated_products = []
        with lock:
            mols_already_computed = [ligand_dictionary[key]["mol"] for key in ligand_dictionary.keys()]
        for info in childes_info:
            mol = AllChem.AddHs(AllChem.MolFromSmiles(info[0]))
            if self.validate_product(mol, mols_already_computed, 0.005):
                ext_info = [mol, info[0], info[1], info[2]]
                validated_products.append(ext_info)
        validated_products = np.array(validated_products)
        print("Generated :" + str(len(validated_products)) + " childes")
        if len(validated_products) == 0:
            return
        clusters = DrugDiscoverySupervisor.__cluster_fps(validated_products[:, 0], 0.005)
        centers = [clusters[i][0] for i in range(len(clusters))]
        validated_products = validated_products[centers]
        print("Clustered in :" + str(len(validated_products)) + " childes")

        folder = os.path.join(self.working_dir, "Mutation_" + parent_1.id)

        if not os.path.exists(folder):
            os.mkdir(folder)
            os.system("cp " + self.protein_pdb + " " + os.path.join(folder, self.protein_pdb))

        mfee = MFEESampler(working_directory=folder,
                           protein_pdb=os.path.join(folder, self.protein_pdb),
                           log_file=os.path.join(folder, "log_Mutation_" + parent_1.id + ".dat"),
                           learning_rate=0.02, embed_dimension=self.embed_dimension, n_threads=self.n_threads,
                           domain_selection=self.domain_selection, device="cpu")
        smiles = validated_products[:, 1]
        names = []
        info = []
        print(validated_products)
        with lock:
            for k in range(len(validated_products)):
                with compound_number.get_lock():
                    names.append("compound_" + str(compound_number.value))
                    compound_number.value += 1
                # TODO find a way to get the smile of the compound
                if validated_products[k][3] is not None:
                    parent_2 = Ligand.Ligand_info(self.zinc_dictionary[validated_products[k][3]], parent_1.name,
                                                  validated_products[k][3])
                else:
                    parent_2 = Ligand.Ligand_info(parent_1.smile, parent_1.name,
                                                  "1_reactant")
                info.append([parent_1, parent_2, validated_products[k][2]])

        mfee.load_info(info)
        mfee.load_chemical_space(smiles, names)
        mfee.load_vae(file=mfee_file)
        print("Starting iterations")
        for i in range(n_iterations):
            mfee.step(smiles, i)
            # adjourn the shared dictionary
            with lock:
                mfee.adjourn_dictionary(ligand_dictionary)
                smiles_train = []
                dg_train = []
                for key in ligand_dictionary.keys:
                    dg_train.append(ligand_dictionary[key]["dg"])
                    smiles_train.append(ligand_dictionary[key]["smiles"])
                for s in range(len(self.train_smiles_dg)):
                    if self.train_smiles_dg[s][0] not in smiles_train:
                        smiles_train.append(self.train_smiles_dg[s][0])
                        dg_train.append(self.train_smiles_dg[s][1])
                mfee.train_network(smiles_train, dg_train, 100, file=os.path.join(folder, "mfee_mutation_100epochs.pth"))

                if i >= 5:
                    best_names = self.__get_best_dg(ligand_dictionary, n_best=int(0.3 * len(ligand_dictionary.keys())))
                    # if the job is not giving results terminate
                    kill_me = True
                    my_keys = mfee.computed_dictionary.keys()
                    for key in my_keys:
                        if key in best_names:
                            kill_me = False
                    if kill_me:
                        # messanger.put(["KILL", process_id])
                        return

        # ask Watchdog to terminate the process
        # messanger.put(["KILL", process_id])

    def save_dictionary(self, dictionary, file):
        dictionary_1 = {}
        for key in dictionary.keys():
            dictionary_1[key] = {"dg": dictionary[key]["dg"],
                                 "name": dictionary[key]["name"],
                                 "smiles": dictionary[key]["smiles"]}
        jn = json.dumps(dictionary_1)
        f = open(file, "w")
        f.write(jn)
        f.close()

    def __get_best_dg(self, dictionary, n_best):
        names = []
        dg = []

        for ligand_name in dictionary.keys():
            names.append(dictionary[ligand_name]["name"])
            dg.append(dictionary[ligand_name]["dg"])
        dg = np.array(dg)
        names = np.array(names)
        idx = np.argsort(dg)
        return names[idx][:n_best], dg[idx][:n_best]

    def run(self, n_iterations, crossover_range, vfee_min_training_data=0):
        self.watchdog = DrugDiscoverySupervisor.Watchdog(self.n_max_parallel_process)
        self.vfee_min_training_data = vfee_min_training_data

        smiles_all = [AllChem.MolToSmiles(AllChem.RemoveHs(self.zinc_dictionary[m])) for m in
                      self.zinc_dictionary.keys()]

        if not os.path.exists(self.fragments_directory):
            os.system("mkdir " + self.fragments_directory)

        os.system("cp " + self.protein_pdb + " " + os.path.join(self.fragments_directory, self.protein_pdb))

        vae = VAESampler(working_directory=self.fragments_directory,
                         protein_pdb=os.path.join(self.fragments_directory,
                                                  self.protein_pdb),
                         log_file=os.path.join(self.fragments_directory,
                                               "log_ZINC_library.dat"),
                         learning_rate=0.05, embed_dimension=self.embed_dimension, n_threads=self.n_threads,
                         domain_selection=self.domain_selection)
        self.vae_sampler = vae
        self.mfee_sampler.load_chemical_space([self.zinc_dictionary[i] for i in self.zinc_dictionary],
                                              [i for i in self.zinc_dictionary])
        vae.load_chemical_space([self.zinc_dictionary[i] for i in self.zinc_dictionary],
                                [i for i in self.zinc_dictionary])

        vae_file = os.path.join(self.working_dir, "vae_100epochs.pth")
        mfee_file = os.path.join(self.working_dir, "mfee_100epochs.pth")
        if not os.path.exists(vae_file):
            print("starting the training")
            vae.train_network(file=vae_file)
            vae.save_vae(vae_file)
        else:
            vae.load_vae(vae_file)
        if os.path.exists(mfee_file):
            self.mfee_sampler.load_vae(mfee_file)

        embed_coords = vae.encode_smiles(smiles_all)
        with open("test.txt", "w") as f:
            for co in embed_coords:
                for c in co:
                    f.write(str(c) + " ")
                f.write("\n")
        prec_point_idx = 10  # np.random.randint(0, len(smiles_all) - 1)
        manager = Manager()
        ligands_dictionary = manager.dict()
        lock = mp.Lock()
        compound_number = mp.Value('i', 0)
        best_n_ligands_names = []
        n_processes = 0

        # define the optimizer for the MFEE
        # optimizer = optim.Adam(self.mfee.parameters())
        for i in range(n_iterations):
            if i < vfee_min_training_data:
                try:
                    prec_point_idx = vae.step(embed_coords, prec_point_idx, i)
                except:
                    prec_point_idx = np.random.randint(0, len(smiles_all) - 1)
                with lock:
                    vae.adjourn_dictionary(ligands_dictionary)
            else:
                self.mfee_sampler.step(smiles_all, i)

                dg = []
                names = []
                smiles_temp = []

                # adjourn the dictionary
                with lock:
                    self.mfee_sampler.adjourn_dictionary(ligands_dictionary)
                    self.save_dictionary(ligands_dictionary, os.path.join(self.working_dir, "dict.json"))

                    for ligand_name in ligands_dictionary.keys():
                        names.append(ligands_dictionary[ligand_name]["name"])
                        dg.append(ligands_dictionary[ligand_name]["dg"])
                        smiles_temp.append(ligands_dictionary[ligand_name]["ligand"].smile)
                for s in range(len(self.train_smiles_dg)):
                    if self.train_smiles_dg[s][0] not in smiles_temp:
                        smiles_temp.append(self.train_smiles_dg[s][0])
                        dg.append(self.train_smiles_dg[s][1])
                smiles_temp = np.array(smiles_temp)
                dg = np.array(dg)

                self.mfee_sampler.train_network(smiles_temp, dg, 100, batch_size=4, noise=0.2,
                                                file=os.path.join(self.working_dir, "mfee.pth"))
                for g in range(len(smiles_temp)):
                    comp_dg = self.mfee_sampler.estimate_free_energy(smiles_temp[g])
                    print("{} {}".format(comp_dg, dg[g]))

            if i >= self.lag_iterations:
                dg = []
                names = []
                smiles_temp = []

                # adjourn the dictionary
                print(vae.computed_dictionary)
                print(self.mfee_sampler.computed_dictionary)

                with lock:
                    vae.adjourn_dictionary(ligands_dictionary)
                    self.mfee_sampler.adjourn_dictionary(ligands_dictionary)

                    for ligand_name in ligands_dictionary.keys():
                        names.append(ligands_dictionary[ligand_name]["name"])
                        dg.append(ligands_dictionary[ligand_name]["dg"])
                        smiles_temp.append(ligands_dictionary[ligand_name]["ligand"].smile)

                dg = np.array(dg)
                names = np.array(names)
                idx = np.argsort(dg)
                print(ligands_dictionary.keys())
                names = names[idx][:crossover_range]
                new_job = False
                parent_1_name = None
                for k in range(len(names)):
                    if names[k] not in best_n_ligands_names:
                        new_job = True
                        parent_1_name = names[k]
                        break
                if new_job:
                    best_n_ligands_names = names
                    parent_1 = ligands_dictionary[parent_1_name]["ligand"]
                    self.watchdog.submit_process(self.__mutation_worker, (parent_1,
                                                                          int(n_iterations / 2),
                                                                          vae_file,
                                                                          n_processes,
                                                                          ligands_dictionary,
                                                                          compound_number,
                                                                          lock,
                                                                          self.watchdog.queue,))
                    print(len(self.watchdog.waiting_processes), len(self.watchdog.processes))
                    n_processes += 1

