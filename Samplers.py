from abc import ABC
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn import manifold
from Utility import *
from numpy import random
import multiprocessing
from torch import optim
from VAE.featurizer import *
from VAE.models import *
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Score(ABC):
    working_dir = None

    def set_workingdir(self, working_dir):
        pass

    def valuate(self, protein_pdb, ligands, domain_selection=None):
        pass


class Sampler(ABC):
    def find_minimum(self, n_iterations):
        pass

    def reset(self):
        pass

    def report(self, string):
        with open(self.log_file, "a") as file:
            file.write(string + "\n")

    def load_chemical_space(self, input_mol, names=None):
        self.names = names
        self.smiles = []
        for mol in input_mol:
            molec = None
            if isinstance(mol, AllChem.Mol):
                molec = mol
            elif isinstance(mol, str):
                if ".pdb" in mol:
                    if os.path.exists(mol):
                        molec = AllChem.MolFromPDBFile(mol)
                else:
                    try:
                        molec = AllChem.MolFromSmiles(mol)
                    except:
                        try:
                            molec = AllChem.MolFromInchi(mol)
                        except:
                            print('Input molecule format not excepted')

            self.molecules.append(molec)
            self.smiles.append(AllChem.MolToSmiles(molec))


class MontecarloSampler(Sampler):
    def __init__(self, working_directory, protein_pdb, log_file):
        self.molecules = []
        self.working_dir = working_directory
        self.protein_pdb = protein_pdb
        self.log_file = log_file

    def reset(self):
        self.molecules.clear()

    def find_minimum(self, n_iterations):
        computed_i = []
        ligands = []
        ligands_names = []
        for i in range(n_iterations):

            # Choose a random point and evaluate
            while True:
                rand_i = np.random.randint(0, len(self.molecules))
                if rand_i not in computed_i:
                    break

            ligands.append(self.molecules[rand_i])
            ligands_names.append(str(rand_i))
        print(ligands_names)
        score = Autodock(os.path.join(self.working_dir, "out_autodock.dat"), folder=self.working_dir)
        dg = score.valuate(self.protein_pdb, ligands, ligands_names)
        for g in range(len(dg)):
            self.report(ligands_names[g] + " " + str(dg[g]))


class DGDSampler(Sampler):
    def __init__(self, working_directory, protein_pdb, log_file, learning_rate=0.05, embed_dimension=2, n_threads=1,
                 domain_selection=None):
        self.molecules = []
        self.working_dir = working_directory
        self.protein_pdb = protein_pdb
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.n = embed_dimension
        self.n_threads = n_threads
        self.domain_selection = domain_selection

    def reset(self):
        self.molecules.clear()

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

    @staticmethod
    def _subdivide_array(a, n):
        """
        Subdivide the array a in n parts
        :param a: input array
        :param n: number of subdivisions
        :return: n array
        """
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

    def transorm_Auto_Encoders(self, molecules, n_dimensions):
        """
        Perform a  Auto-Encoders scaling over a set of molecules. It will return the coordinates in the new base
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
    def get_nearest_point(idx, coords, n, exclusion_list=[]):
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

    def find_minimum(self, n_iterations):
        """
        Perform a gradient descendant in  n_iteration,
        :param n_iterations:      Number of iteration for each generation
        :return:
        """

        # compute the coordinates in the reduced space
        embed_coords = self.transorm_MDS(self.molecules, self.n)

        # choose a molecule to perform the coordinate change
        prec_point = random.randint(0, len(self.molecules) - 1)

        all_dg = []
        all_ligands_idx = []

        for it in range(n_iterations):

            # choose as coordinates points the nearest to prec_point not computed yet
            idx = self.get_nearest_point(prec_point, embed_coords, self.n, all_ligands_idx)

            if len(idx) == 0:
                break

            # compute the score of those molecules
            idx.append(prec_point)
            score = Autodock(os.path.join(self.working_dir, "out_autodock.dat"),
                             folder=self.working_dir,
                             n_threads=self.n_threads)
            ligands = np.array(self.molecules)[idx]
            if self.names is None:
                ligands_names = [str(i) for i in idx]
            else:
                ligands_names = [self.names[i] for i in idx]

            # translate all the coordinates: center X_0 (precedent point), X -> X' = X - X_0
            x_0 = prec_point
            translate = np.array([embed_coords[x_0]] * len(embed_coords))
            embed_coords_tr = embed_coords - translate

            # create the matrix of change base AX'' = X' and compute the inverse X'' = A^(-1)(X') = A^(-1)(X - X_0)
            matrix = embed_coords_tr.transpose()[:, idx[:(len(idx) - 1)]].transpose()
            for l in range(len(matrix)):
                matrix[l] /= np.sqrt(np.dot(matrix[l], matrix[l]))

            matrix_inv = np.linalg.inv(matrix)

            embed_coords_new = np.array([np.matmul(x, matrix_inv) for x in embed_coords_tr])

            dg = score.valuate(self.protein_pdb, ligands, ligands_names, domain_selection=self.domain_selection)
            dg_prec = dg[len(dg) - 1]
            all_dg.extend(dg)
            all_ligands_idx.extend(idx)

            self.report("#iteration " + str(it))
            for s in range(len(dg)):
                coord = embed_coords[idx[s]]
                st = ""
                for c in coord:
                    st += str(c) + " "
                st += str(dg[s])
                self.report(st)

            if len(idx) < self.n:
                break

            # compute the gradient
            df = np.array([0.0] * self.n)
            for m in range(0, self.n):
                dx = np.sqrt(np.dot(embed_coords_tr[idx[m]],
                                    embed_coords_tr[idx[m]]))
                df[m] = (dg[m] - dg_prec) / dx

            print("Gradient X': ")
            print(df)
            df = np.matmul(matrix, df)
            print("Gradient: ")
            print(df)

            # compute the next point in X coordinates: X_t+1 = X_t - lr grad f(X_0)
            next_coords = - self.learning_rate * df + embed_coords[x_0]

            # look for the nearest point not computed yet
            min_dx = np.sqrt(np.dot(next_coords - embed_coords[0], next_coords - embed_coords[0]))
            min_b = 0
            for b in range(1, len(embed_coords)):
                dx = np.sqrt(np.dot(next_coords - embed_coords[b], next_coords - embed_coords[b]))
                if (dx < min_dx) and (b not in all_ligands_idx):
                    min_dx = dx
                    min_b = b

            prec_point = min_b

        all_dg = np.array(all_dg)
        indx = np.argsort(all_dg)
        dg = all_dg[indx]
        all_ligands_idx = np.array(all_ligands_idx)
        ligands = all_ligands_idx[indx]
        return ligands, dg


class Autodock(Score):
    def __init__(self, output_file=None, folder='input/', keep_ions=True, open_mode='w', n_threads=1):
        self.output_file = output_file
        self.folder = folder
        self.keep_ions = keep_ions
        self.open_mode = open_mode
        self.n_threads = n_threads
        self.ligands = []

    def set_workingdir(self, working_dir):
        self.working_dir = working_dir

    def __generate_ligands(self, th_id, from_i, to_i, ligands, ligands_name, ligands_folder, out):
        if not os.path.exists("THREAD_" + str(th_id)):
            os.mkdir("THREAD_" + str(th_id))
        os.chdir("THREAD_" + str(th_id))
        print("running thread " + str(th_id))
        for i in range(from_i, to_i):
            print("generating ligand " + ligands_name[i])
            out[i] = Ligand(ligands[i], ligands_name[i], ligands_folder[i])
        os.chdir("../")
        # os.system("rm -r THREAD_" + str(th_id))

    def valuate(self, protein_pdb, ligands, ligands_names=None, domain_selection=None, ligands_info=None):
        lig_name = 'LIG'
        name = protein_pdb.split('/')[len(protein_pdb.split('/')) - 1].split('.')[0]
        protein = Protein(protein_pdb, name, True, self.folder)
        dG = []

        if not os.path.exists(self.folder + "/ligands"):
            os.mkdir(self.folder + "/ligands")
        if not os.path.exists(self.folder + "/complexes"):
            os.mkdir(self.folder + "/complexes")

        self.ligands = [None] * len(ligands)
        ligands_name = []
        ligands_folder = []
        for k in range(len(ligands)):
            if ligands_names is not None:
                ligand_folder = os.path.join(self.folder, "ligands", ligands_names[k])
            else:
                ligand_folder = self.folder + "/ligands/" + str(k)
            ligands_name.append(lig_name)
            ligands_folder.append(ligand_folder)

        # pool = multiprocessing.Pool(processes=len(ligands)) pool.map(target=self.__generate_ligands, [(i,
        # ligands[i], ligands_name[i], ligands_folder[i], self.ligands[i],) for i in range(len(ligands))])
        # TODO Use a Pool object

        threads = [multiprocessing.Process(target=self.__generate_ligands,
                                           args=(i,
                                                 int(i * len(ligands) / self.n_threads),
                                                 int((i + 1) * len(ligands) / self.n_threads),
                                                 ligands, ligands_name, ligands_folder, self.ligands,))
                   for i in range(self.n_threads)]

        for th in threads:
            th.start()

        for th in threads:
            th.join()

        ligands_computed = []
        for k in range(len(ligands)):
            try:
                output = open(self.output_file, 'a')
                if ligands_names is not None:
                    id = ligands_names[k]

                ligand = Ligand(ligands[k], lig_name, ligands_folder[k], id=id)
                ligands_computed.append(ligand)
                if ligands_info is not None:
                    ligand.set_info(ligands_info[k][0], ligands_info[k][1], ligands_info[k][2])

                if ligands_names is not None:
                    complex_folder = os.path.join(self.folder, "complexes", ligands_names[k])
                    complex_name = protein.name + "_" + ligands_names[k]
                else:
                    complex_folder = os.path.join(self.folder, "complexes", str(k))
                    complex_name = name + "_" + str(k)

                _complex = Complex(protein, ligand, complex_name,
                                   working_dir=complex_folder,
                                   domain_selection=domain_selection)

                output.write(str(k) + " " + ligand.smile + " " + str(_complex.dG_autodock) + "\n")
                output.close()
                dG.append(_complex.dG_autodock)
            except:
                print("error valutate autodock")
        return dG, ligands_computed


class VAESampler(Sampler):
    def __init__(self, working_directory, protein_pdb, log_file, learning_rate=0.05, embed_dimension=3, n_threads=1,
                 domain_selection=None, device=None):
        self.molecules = []
        self.working_dir = working_directory
        self.protein_pdb = protein_pdb
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.n = embed_dimension
        self.n_threads = n_threads
        self.domain_selection = domain_selection
        self.vae = MolecularVAE(self.n, device)
        self.computed_idx = []
        self.computed_dg = []
        self.computed_dictionary = {}
        self.info = None

    def reset(self):
        self.molecules.clear()
        self.computed_idx = []
        self.computed_dg = []

    def train_network(self, epochs=100, file=None):
        def loss_function(recon_x, x, mu, logvar):
            BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD

        smiles = []  # [AllChem.MolToSmiles(AllChem.RemoveHs(m)) for m in self.molecules]
        for m in self.molecules:
            s = AllChem.MolToSmiles(AllChem.RemoveHs(m)).ljust(120)
            if len(s) <= 120:
                smiles.append(s)

        ohf = OneHotFeaturizer()
        temp = []
        for i in range(len(smiles)):
            t = torch.from_numpy(ohf.featurize([smiles[i]]))[0].float()
            temp.append(t)
            if i % 100 == 0:
                print(i)
        train = train = torch.stack(temp)
        train = torch.utils.data.TensorDataset(train)
        train_loader = torch.utils.data.DataLoader(train, batch_size=250, shuffle=True)

        optimizer = optim.Adam(self.vae.parameters())

        self.vae.train()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                data = data[0].to(self.vae.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'{epoch} / {batch_idx}\t{loss:.4f}')
            if file is not None:
                self.save_vae(file)
            print('train', train_loss / len(train))

    def save_vae(self, file="net_weight.pth"):
        torch.save(self.vae.state_dict(), file)

    def load_vae(self, file="net_weight.pth"):
        self.vae.load_state_dict(torch.load(file, map_location=self.vae.device))

    def encode_smiles(self, smiles):
        res = []
        for s in smiles:
            start = s.ljust(120)
            oh = OneHotFeaturizer()
            start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to(self.vae.device)
            res.append(self.vae.encode(start_vec)[0].cpu().detach().numpy()[0])
        return res

    def adjourn_dictionary(self, dictionary):
        keys = list(dictionary.keys())
        for key in self.computed_dictionary.keys():
            if key not in keys:
                dictionary[key] = copy.deepcopy(self.computed_dictionary[key])

    def load_info(self, info):
        self.info = np.array(info)

    def step(self, embed_coords, prec_point_idx, iteration_n):
        # compute the coordinates in the reduced space
        # embed_coords = self.encode_smiles(self.smiles, self.n)

        # choose as coordinates points the nearest to prec_point not computed yet
        idx = DGDSampler.get_nearest_point(prec_point_idx, embed_coords, self.n, self.computed_idx)

        if len(idx) == 0:
            print("No data to compute")
            return None

        # compute the score of those molecules
        idx.append(prec_point_idx)
        score = Autodock(os.path.join(self.working_dir, "out_autodock.dat"),
                         folder=self.working_dir,
                         n_threads=self.n_threads)
        ligands = np.array(self.molecules)[idx]
        if self.names is None:
            ligands_names = [str(i) for i in idx]
        else:
            ligands_names = [self.names[i] for i in idx]

        # translate all the coordinates: center X_0 (precedent point), X -> X' = X - X_0
        x_0 = prec_point_idx
        translate = np.array([embed_coords[x_0]] * len(embed_coords))
        embed_coords_tr = embed_coords - translate

        # create the matrix of change base AX'' = X' and compute the inverse X'' = A^(-1)(X') = A^(-1)(X - X_0)
        matrix = embed_coords_tr.transpose()[:, idx[:(len(idx) - 1)]].transpose()
        for l in range(len(matrix)):
            matrix[l] /= np.sqrt(np.dot(matrix[l], matrix[l]))

        matrix_inv = np.linalg.inv(matrix)

        embed_coords_new = np.array([np.matmul(x, matrix_inv) for x in embed_coords_tr])

        if self.info is not None:
            dg, ligands_class = score.valuate(self.protein_pdb, ligands, ligands_names,
                                              domain_selection=self.domain_selection,
                                              ligands_info=self.info[idx])
        else:
            dg, ligands_class = score.valuate(self.protein_pdb, ligands, ligands_names,
                                              domain_selection=self.domain_selection)
        dg_prec = dg[len(dg) - 1]
        self.computed_dg.extend(dg)
        self.computed_idx.extend(idx)
        for k in range(len(ligands_names)):
            self.computed_dictionary[ligands_names[k]] = {"dg": dg[k], "idx": idx[k], "mol": copy.deepcopy(ligands[k]),
                                                          "prefix": Ligand.get_prefix(ligands_names[k],
                                                                                      self.working_dir),
                                                          "name": ligands_names[k], "ligand": ligands_class[k],
                                                          "smiles": ligands_class[k].smile}

        self.report("#iteration " + str(iteration_n))
        for s in range(len(dg)):
            coord = embed_coords[idx[s]]
            st = ""
            for c in coord:
                st += str(c) + " "
            st += str(dg[s])
            self.report(st)

        # compute the gradient
        df = np.array([0.0] * self.n)
        for m in range(0, self.n):
            dx = np.sqrt(np.dot(embed_coords_tr[idx[m]],
                                embed_coords_tr[idx[m]]))
            df[m] = (dg[m] - dg_prec) / dx

        print("Gradient X': ")
        print(df)
        df = np.matmul(matrix, df)
        print("Gradient: ")
        print(df)

        # compute the next point in X coordinates: X_t+1 = X_t - lr grad f(X_0)
        next_coords = - self.learning_rate * df + embed_coords[x_0]
        print("Next point: ")
        print(next_coords)

        # look for the nearest point not computed yet
        min_dx = np.sqrt(np.dot(next_coords - embed_coords[0], next_coords - embed_coords[0]))
        min_b = 0
        for b in range(1, len(embed_coords)):
            dx = np.sqrt(np.dot(next_coords - embed_coords[b], next_coords - embed_coords[b]))
            if (dx < min_dx) and (b not in self.computed_idx):
                min_dx = dx
                min_b = b

        return min_b

    def find_minimum(self, n_iterations):
        """
        Perform a gradient descendant in  n_iteration,
        :param n_iterations:      Number of iteration for each generation
        :return:
        """

        # compute the coordinates in the reduced space
        embed_coords = self.encode_smiles(self.smiles, self.n)

        # choose a molecule to perform the coordinate change
        prec_point = random.randint(0, len(self.molecules) - 1)

        all_dg = []
        all_ligands_idx = []

        for it in range(n_iterations):

            # choose as coordinates points the nearest to prec_point not computed yet
            idx = DGDSampler.get_nearest_point(prec_point, embed_coords, self.n, all_ligands_idx)

            if len(idx) == 0:
                break

            # compute the score of those molecules
            idx.append(prec_point)
            score = Autodock(os.path.join(self.working_dir, "out_autodock.dat"),
                             folder=self.working_dir,
                             n_threads=self.n_threads)
            ligands = np.array(self.molecules)[idx]
            if self.names is None:
                ligands_names = [str(i) for i in idx]
            else:
                ligands_names = [self.names[i] for i in idx]

            # translate all the coordinates: center X_0 (precedent point), X -> X' = X - X_0
            x_0 = prec_point
            translate = np.array([embed_coords[x_0]] * len(embed_coords))
            embed_coords_tr = embed_coords - translate

            # create the matrix of change base AX'' = X' and compute the inverse X'' = A^(-1)(X') = A^(-1)(X - X_0)
            matrix = embed_coords_tr.transpose()[:, idx[:(len(idx) - 1)]].transpose()
            for l in range(len(matrix)):
                matrix[l] /= np.sqrt(np.dot(matrix[l], matrix[l]))

            matrix_inv = np.linalg.inv(matrix)

            embed_coords_new = np.array([np.matmul(x, matrix_inv) for x in embed_coords_tr])

            dg = score.valuate(self.protein_pdb, ligands, ligands_names, domain_selection=self.domain_selection)
            dg_prec = dg[len(dg) - 1]
            all_dg.extend(dg)
            all_ligands_idx.extend(idx)

            self.report("#iteration " + str(it))
            for s in range(len(dg)):
                coord = embed_coords[idx[s]]
                st = ""
                for c in coord:
                    st += str(c) + " "
                st += str(dg[s])
                self.report(st)

            if len(idx) < self.n:
                break

            # compute the gradient
            df = np.array([0.0] * self.n)
            for m in range(0, self.n):
                dx = np.sqrt(np.dot(embed_coords_tr[idx[m]],
                                    embed_coords_tr[idx[m]]))
                df[m] = (dg[m] - dg_prec) / dx

            print("Gradient X': ")
            print(df)
            df = np.matmul(matrix, df)
            print("Gradient: ")
            print(df)

            # compute the next point in X coordinates: X_t+1 = X_t - lr grad f(X_0)
            next_coords = - self.learning_rate * df + embed_coords[x_0]
            print("Next point: ")
            print(next_coords)

            # look for the nearest point not computed yet
            min_dx = np.sqrt(np.dot(next_coords - embed_coords[0], next_coords - embed_coords[0]))
            min_b = 0
            for b in range(1, len(embed_coords)):
                dx = np.sqrt(np.dot(next_coords - embed_coords[b], next_coords - embed_coords[b]))
                if (dx < min_dx) and (b not in all_ligands_idx):
                    min_dx = dx
                    min_b = b

            prec_point = min_b

        all_dg = np.array(all_dg)
        indx = np.argsort(all_dg)
        dg = all_dg[indx]
        all_ligands_idx = np.array(all_ligands_idx)
        ligands = all_ligands_idx[indx]
        return ligands, dg


class MFEESampler(Sampler):
    def __init__(self, working_directory, protein_pdb, log_file, learning_rate=0.05, embed_dimension=3, n_threads=1,
                 domain_selection=None, device=None):
        self.molecules = []
        self.working_dir = working_directory
        self.protein_pdb = protein_pdb
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.n = embed_dimension
        self.n_threads = n_threads
        self.domain_selection = domain_selection
        self.mfee = MolecularFEE(self.n, device)
        self.optimizer = optim.Adam(self.mfee.parameters())
        self.computed_idx = []
        self.computed_dg = []
        self.computed_dictionary = {}
        self.info = None

    def reset(self):
        self.molecules.clear()
        self.computed_idx = []
        self.computed_dg = []

    class PrepareData(Dataset):

        def __init__(self, X, y, noise=0.0):
            if not torch.is_tensor(X):
                self.X = torch.from_numpy(X).float()
            else:
                self.X = X.float()
            if not torch.is_tensor(y):
                self.y = torch.from_numpy(y).float()
            else:
                self.y = y.float()
            self.noise = noise

        def __len__(self):
            return self.X.size()

        def __getitem__(self, idx):
            noise = (torch.from_numpy(np.array(np.random.randn())) * self.noise).float()
            return self.X[idx], self.y[idx] + noise

    def train_network(self, data_smiles, expected_dg, epochs, noise=0.3, batch_size=4, file=None):
        smiles = []
        for m in data_smiles:
            s = m.ljust(120)
            if len(s) <= 120:
                smiles.append(s)
        temp = []
        ohf = OneHotFeaturizer()
        for i in range(len(smiles)):
            t = torch.from_numpy(ohf.featurize([smiles[i]]))[0].float()
            temp.append(t)
        train = torch.stack(temp)
        ds = self.PrepareData(train, y=expected_dg, noise=noise)
        train_set = DataLoader(ds, batch_size=batch_size, sampler=SubsetRandomSampler(list(range(len(temp)))))
        train_loss = 0
        loss_function = nn.MSELoss()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            for batch_idx, (data, exp_dg) in enumerate(train_set):
                data = data.to(self.mfee.device)
                exp_dg = exp_dg.to(self.mfee.device)
                self.optimizer.zero_grad()
                out = self.mfee(data)
                out = out.reshape((out.shape[0]))
                loss = loss_function(out, exp_dg)
                loss.backward()
                train_loss += loss
                self.optimizer.step()

        print('train', train_loss.item() / len(temp))
        if file is not None:
            self.save_vae(file)

    def save_vae(self, file="net_weight.pth"):
        torch.save(self.mfee.state_dict(), file)

    def load_vae(self, file="net_weight.pth"):
        self.mfee.load_state_dict(torch.load(file, map_location=self.mfee.device))

    def estimate_free_energy(self, smiles):
        start = smiles.ljust(120)
        oh = OneHotFeaturizer()
        start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to(self.mfee.device)
        res = self.mfee.forward(start_vec)[0].cpu().detach().numpy()
        return res.item()

    def adjourn_dictionary(self, dictionary):
        keys = list(dictionary.keys())
        for key in self.computed_dictionary.keys():
            if key not in keys:
                dictionary[key] = copy.deepcopy(self.computed_dictionary[key])

    def load_info(self, info):
        self.info = np.array(info)

    @staticmethod
    def get_idx(idx, already_computed_idx, n):
        idx_clean = []
        for i in range(len(idx)):
            if idx[i] not in already_computed_idx:
                if len(idx_clean) < n:
                    idx_clean.append(idx[i])
                else:
                    break
        return idx_clean

    def step(self, smiles, iteration_n):
        # compute all the smiles score and return the best embed_coords + 1
        dg_extimated = []
        for s in smiles:
            dg_extimated.append(self.estimate_free_energy(s))
        dg_extimated = np.array(dg_extimated)
        idx_sorted = np.argsort(dg_extimated)
        idx = self.get_idx(idx_sorted, self.computed_idx, self.n + 1)
        print(dg_extimated[idx])
        idx = np.array(idx)
        print(idx)

        if len(idx) == 0:
            print("No data to compute")
            return None

        # compute the score of those molecules
        score = Autodock(os.path.join(self.working_dir, "out_autodock.dat"),
                         folder=self.working_dir,
                         n_threads=self.n_threads)
        ligands = np.array(smiles)[idx]
        if self.names is None:
            ligands_names = [str(i) for i in idx]
        else:
            ligands_names = [self.names[i] for i in idx]

        if self.info is not None:
            dg, ligands_class = score.valuate(self.protein_pdb, ligands, ligands_names,
                                              domain_selection=self.domain_selection,
                                              ligands_info=self.info[idx])
        else:
            dg, ligands_class = score.valuate(self.protein_pdb, ligands, ligands_names,
                                              domain_selection=self.domain_selection)
        self.computed_dg.extend(dg)
        self.computed_idx.extend(idx)
        for k in range(len(ligands_names)):
            self.computed_dictionary[ligands_names[k]] = {"dg": dg[k], "idx": idx[k],
                                                          "mol": ligands_class[k].Mol,
                                                          "prefix": Ligand.get_prefix(ligands_names[k],
                                                                                      self.working_dir),
                                                          "name": ligands_names[k],
                                                          "ligand": ligands_class[k],
                                                          "smiles": ligands_class[k].smile}

        self.report("#iteration " + str(iteration_n))

        for c in range(len(dg)):
            self.report(ligands_names[c] + " " + ligands_class[c].smile + " " + str(dg[c]))

        return 0
