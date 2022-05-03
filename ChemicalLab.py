import threading as thread
from FAT import *
from autogrow.Operators.Filter.Filter_classes.FilterClasses.Lipinski_Lenient import *


class VirtualScreeningEnviroment:

    def __init__(self, working_directory, protein_pdb, log_file, n_child=1000, n_threads=1, domain_selection=None):
        self.n_child = n_child
        self.working_dir = working_directory
        self.n_threads = n_threads
        self.protein_pdb = protein_pdb
        self.childes = []
        self.childes_smiles = []
        self.childes_info = []
        self.log_file = log_file
        self.domain_selection = domain_selection

    def report(self, line):
        with open(self.log_file, "a") as file:
            for i in line:
                file.write(str(i) + " ")
            file.write("\n")

    @staticmethod
    def _cluster_fps(ms, cutoff=0.2):
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
    def _fingerprint_similarity(mol_1, mol_2):
        """ returns the calculated similarity between two fingerprints,
              handles any folding that may need to be done to ensure that they
              are compatible

        """
        return DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_1), Chem.RDKFingerprint(mol_2))

    def _react(self, parent_smile, n_child, i_d):

        smileClick = SmilesClickChem(["ClickChem", "", "", ""], [], {})
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
    def _subdivide_array(a, n):
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

    def new_generation(self, parent_smile, clusterize_threshold=0.05):
        """
        Generate a new generation of ligands with all the possible products of all ClickChem reactions with the
        parent molecula
        :param parent_smile:          The SMILES of the molecula to react
        :param clusterize_threshold:  elements within this range of each other are considered
                                      to be neighbors
        :return:                      The Molecules that represent the clustered products of all the possible
                                      reactions with the parent molecule
        """
        self.childes = []
        all_childes_info = []

        for s in range(len(parent_smile)):

            self.childes_smiles = [[] for i in range(self.n_threads)]
            self.childes_info = [[] for i in range(self.n_threads)]

            threads = [thread.Thread(target=self._react,
                                     args=(parent_smile[s], self.n_child, i,))
                       for i in range(self.n_threads)]

            for th in threads:
                th.start()

            for th in threads:
                th.join()
            for i in range(0, self.n_threads):

                for k in range(len(self.childes_info[i])):
                    self.childes_info[i][k].append(" parent: " + str(s))

                all_childes_info.extend(self.childes_info[i])

                for j in range(0, len(self.childes_smiles[i])):
                    mol = AllChem.MolFromSmiles(self.childes_smiles[i][j])
                    AllChem.AddHs(mol)
                    res = AllChem.EmbedMolecule(mol)
                    if res == 0:
                        valid = True
                        for a in mol.GetConformers():
                            coords = a.GetPositions()
                            for c in coords:
                                valid = valid and (c[0] != 0.0) and (c[1] != 0.0) and (c[2] != 0.0)

                        if valid:
                            self.childes.append(mol)

            print("generated " + str(len(self.childes)) + " childes\n")
        self.childes = np.array(self.childes)

        clusters = VirtualScreeningEnviroment._cluster_fps(self.childes, clusterize_threshold)
        centers = [clusters[i][0] for i in range(len(clusters))]
        clustered_childes = self.childes[centers]
        all_childes_info = np.array(all_childes_info)
        clustered_childes_info = all_childes_info[centers]

        print("clustered in  " + str(len(clustered_childes)) + " centers\n")

        for c in range(len(clustered_childes_info)):
            self.report([c, clustered_childes_info[c, 0], clustered_childes_info[c, 1], clustered_childes_info[c, 2],
                         clustered_childes_info[c, 3]])

        return clustered_childes, clustered_childes_info[:, 0]

    def transorm_MDS(self, molecules, n_dimensions):
        """
        Perform a Multidimensional scaling over a set of molecules. First compute the distance matrix, then perform
        MDS. It will return the coordinates in the new base
        :param molecules:     An iterable with the input molecules
        :param n_dimensions:  Number of dimensions in which to immerse.
        :return:              The embedded coordinates
        """
        seed = np.random.RandomState(seed=3)
        nmds = manifold.MDS(n_components=n_dimensions, metric=False, max_iter=3000, eps=1e-12,
                            dissimilarity="precomputed", random_state=seed, n_jobs=self.n_threads,
                            n_init=1)

        # first generate the distance matrix:
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in molecules]
        dists = []
        print("Generate the distance matrix\n")

        for i in range(0, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            dists.append([1 - x for x in sims])

        X = nmds.fit_transform(dists)
        return X

    @staticmethod
    def get_nearest_point(idx, coords, n, exclusion_list):
        """
        Find the nearest n points between all coords at the coordinates coords[idx] that are not in exclusion_list
        :param idx:              The index of the coordinate
        :param coords:           List of all coordinates
        :param n:                Number of points to get
        :param exclusion_list:   Excluded points
        :return:                 list of n, with the indexes in the coords array of the nearest n points
        """
        distances = [np.sqrt(np.dot(coords[p] - coords[idx], coords[p] - coords[idx])) for p in range(0, len(coords))]
        indx = np.argsort(distances)
        res = []
        for i in indx:
            if i not in exclusion_list:
                res.append(i)
        return res[1:n + 1]

    def run_gradient(self, starting_smile, n_steps, n_iterations=100, choose_best_n=3, recover_generation=None):
        """
        Perform a gradient descendant in order to improve the starting smile. From each generation try to find the best
        ligand in n_iteration, then choose the best (choose_best_n) and finally create the next generation
        :param starting_smile:    The SMILES of the ligand to improve
        :param n_steps:           Number of generations
        :param n_iterations:      Number of iteration for each generation
        :param choose_best_n:     Number of parents to use for the the next generation
        :return:
        """

        precedent_generation_smiles = [starting_smile]
        log = open(self.working_dir + "/log.txt", "w")
        starting_gen = 0
        if recover_generation is not None:
            indxes_all = []
            dg_all = []
            all_smiles = []
            if os.path.exists(self.working_dir + "/generation_" + str(recover_generation)):
                with open(self.log_file, "r") as log_pre:
                    for line in log_pre:
                        words = line.split()
                        if len(words) > 3:
                            if os.path.exists(self.working_dir +
                                              "/generation_" + str(recover_generation) +
                                              "/complexes/" + str(words[0]) +
                                              '/log.txt'):
                                dG, rmsd_l = FileUtility.read_log(self.working_dir +
                                                                  "/generation_" + str(recover_generation) +
                                                                  "/complexes/" + str(words[0]) +
                                                                  '/log.txt', line=str(1), col=[1, 2])
                                all_smiles.append(words[1])
                                dg_all.append(dG)
                                indxes_all.append(str(words[0]))

            all_smiles = np.array(all_smiles)
            dg_all = np.array(dg_all)
            indxes_all = np.array(indxes_all)

            indx = np.argsort(dg_all)
            precedent_generation_smiles = all_smiles[indx][:choose_best_n]
            best_dg = dg_all[indx][:choose_best_n]
            best_indxes = indxes_all[indx][:choose_best_n]

            self.report(["######################################"])
            self.report(["############ best ligands ############"])
            self.report(["################# " + str(recover_generation) + " #################"])
            self.report(["######################################"])

            for lig in range(len(best_indxes)):
                self.report([str(best_indxes[lig]), str(best_dg[lig])])

            self.report(["######################################"])

            starting_gen = recover_generation + 1

        for i in range(starting_gen, n_steps):

            if not os.path.exists(self.working_dir + "/generation_" + str(i)):
                os.mkdir(self.working_dir + "/generation_" + str(i))

            new_generation = []

            # generate n_childs for each parent in precedent generation
            # for j in range(len(precedent_generation_smiles)):
            clustered_childes, generation_smiles = self.new_generation(precedent_generation_smiles)
            for child in clustered_childes:
                new_generation.append(child)

            for j in range(0, len(new_generation)):
                AllChem.MolToPDBFile(new_generation[j],
                                     self.working_dir + "/generation_" + str(i) + "/" + str(j) + ".pdb")

            new_generation = np.array(new_generation)
            n_dimensions = self.n_threads - 1

            if len(new_generation) < n_dimensions:
                score = Autodock("out.txt", folder=self.working_dir + "/generation_" + str(i))

                os.system("cp " + self.working_dir + "/" + self.protein_pdb + " " +
                          self.working_dir + "/generation_" + str(i) + "/" + self.protein_pdb)

                dg = score.valuate(self.protein_pdb, new_generation)

                for g in range(len(dg)):
                    self.report([g, dg[g]])

                dg = np.array(dg)
                indx = np.argsort(dg)
                dg = dg[indx]
                ligands = new_generation[indx]
                best_ligands = ligands[:choose_best_n]

                for lig in range(len(best_ligands)):
                    log.write(AllChem.MolToSmiles(AllChem.RemoveHs(best_ligands[lig])) + " " + str(dg[lig]) + "\n")

                precedent_generation_smiles = [AllChem.MolToSmiles(AllChem.RemoveHs(best_ligands[b])) for b in
                                               range(len(best_ligands))]
            else:

                sampler = DGDSampler(working_directory=os.path.join(self.working_dir, "generation_" + str(i)),
                                     protein_pdb=os.path.join(self.working_dir, "generation_" + str(i),
                                                              self.protein_pdb),
                                     log_file=os.path.join(self.working_dir, "generation_" + str(i),
                                                           "log_gen_" + str(i) + ".dat"),
                                     learning_rate=0.05, embed_dimension=n_dimensions, n_threads=self.n_threads,
                                     domain_selection=self.domain_selection)
                sampler.load_chemical_space(new_generation)
                ligands, dg = sampler.find_minimum(n_iterations)

                dg = np.array(dg)
                indx = np.argsort(dg)
                dg = dg[indx]
                ligands = new_generation[indx]
                best_ligands = ligands[:choose_best_n]

                self.report(["######################################"])
                self.report(["############ best ligands ############"])
                self.report(["################# " + str(i) + " #################"])
                self.report(["######################################"])

                for lig in range(len(best_ligands)):
                    self.report([str(indx[lig]), str(dg[lig])])

                self.report(["######################################"])

                precedent_generation_smiles = generation_smiles[indx]

    def novo_design(self, n_steps, n_iterations=100, choose_best_n=3, ):
        ZINC_compounds = []
        path = os.path.join("autogrow", "Operators", "Mutation", "SmileClickChem", "Reaction_libraries", "ClickChem",
                            "complimentary_mol_dir")
        for root, dirs, files in os.walk(path):
            for name in files:
                file = os.path.join(root, name)
                with open(file, "r") as inp:
                    for line in inp:
                        words = line.split()
                        if len(words) >= 2:
                            smile = words[0]
                            ZINC_id = words[1]
                            mol = AllChem.MolFromSmiles(smile)
                            mol = AllChem.AddHs(mol)
                            if AllChem.EmbedMolecule(mol) == 0:
                                ZINC_compounds.append([len(ZINC_compounds), smile, -1, ZINC_id, mol])

        ZINC_compounds = np.array(ZINC_compounds)
        # ZINC_compounds_clusters = self._cluster_fps(ZINC_compounds[:, 4], 0.05)
        # centers = [ZINC_compounds_clusters[i][0] for i in range(len(ZINC_compounds_clusters))]
        print(len(ZINC_compounds))

        precedent_generation_info = np.array(ZINC_compounds)  # [ZINC_compounds[centers, 0]]
        precedent_generation_pdb = []
        for i in range(0, n_steps):

            if not os.path.exists(self.working_dir + "/generation_" + str(i)):
                os.mkdir(self.working_dir + "/generation_" + str(i))

            for info in precedent_generation_info:
                mol = AllChem.MolFromSmiles(info[1])
                mol = AllChem.AddHs(mol)
                if AllChem.EmbedMolecule(mol) == 0:
                    AllChem.MolToPDBFile(mol, os.path.join(self.working_dir, "generation_" + str(i),
                                                           str(info[0]) + ".pdb"))
                    precedent_generation_pdb.append(os.path.join(self.working_dir, "generation_" + str(i),
                                                                 str(info[0]) + ".pdb"))

            os.system("cp " + os.path.join(self.working_dir, self.protein_pdb) + " " + os.path.join(self.working_dir,
                                                                                                    "generation_" + str(
                                                                                                        i),
                                                                                                    self.protein_pdb))

            n_dimensions = self.n_threads

            sampler = DGDSampler(working_directory=os.path.join(self.working_dir, "generation_" + str(i)),
                                 protein_pdb=os.path.join(self.working_dir, "generation_" + str(i),
                                                          self.protein_pdb),
                                 log_file=os.path.join(self.working_dir, "generation_" + str(i),
                                                       "log_gen_" + str(i) + ".dat"),
                                 learning_rate=0.05, embed_dimension=n_dimensions, n_threads=self.n_threads,
                                 domain_selection=self.domain_selection)
            sampler.load_chemical_space(precedent_generation_info[:, 4])
            ligands, dg = sampler.find_minimum(n_iterations)

            dg = np.array(dg)
            indx = np.argsort(dg)
            dg = dg[indx]
            ligands = precedent_generation_info[indx]
            best_ligands = ligands[:choose_best_n]

            # generate n_childs for each parent in precedent generation
            new_generation = []
            for j in range(len(best_ligands)):
                clustered_childes, generation_smiles = self.new_generation(best_ligands[j, 1])
                for child in clustered_childes:
                    new_generation.append(child)
