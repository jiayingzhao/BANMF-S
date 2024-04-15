import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import io as sio
from scipy.sparse import linalg
import multiprocessing as mp
import Mat2BlockPublic as Mat2Block
import math, time, random, os

NON_ZERO_BONUS = 2.5


class BANMFS(object):
    def __init__(self,infolder, outfolder, R, B, k, scheme_flag, alpha1, alpha2, gamma1, gamma2,
                 epsilon_start, epsilon_end, max_iter, seed):

        # hyperparameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.R = R
        self.B = B
        self.k = k

        # flags
        self.sym_flag = 1
        self.shuf_flag = 0
        self.diag_flag = 1
        self.anomaly_which = 'p'
        self.scheme_flag = scheme_flag

        # data info init ------ will be updated in banmfs.ComputeDimension()
        self.num_node = 0
        self.num_attr = 0

        # implementation parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.max_iter = max_iter
        self.check_iter = math.ceil(max_iter/10)

        # data io
        self.infolder = infolder
        self.outfolder = outfolder 

        # error records
        self.err_vec = []

        # seed
        self.seed = seed



    def ReadInput(self):

        """A, X, L: matrices already segmented into blocks(bins)
           A: cell similarity matrix;
           L: gene similarity matrix;
           X: expression matrix
           R: rank, or # groups
           B: # blocks for parallel implementation
        """
        
        self.ProcPath()
        print('Read cell similarity matrix...')
        self.A, self.shufa = Mat2Block.Adjmat2Bins(self.fn_a, self.B,
                                                    self.sym_flag,
                                                    self.shuf_flag,
                                                    self.diag_flag,
                                                    self.seed)


        print('Read gene similarity matrix...')
        self.L, self.shufl = Mat2Block.Adjmat2Bins(self.fn_l, self.B,
                                                    self.sym_flag,
                                                    self.shuf_flag,
                                                    self.diag_flag,
                                                    self.seed)
        
        print('Read expression matrix...')
        self.X, self.M = Mat2Block.X2Bins(self.fn_x, self.B,
                                           self.shuf_flag, self.shufa, self.shufl)
        print('Process computing nodes ...')
        self.ComputeDimension()


    def ChooseR(self, n_start = 79, n_end = 100, thresh=6):
        X = self.X
        U, S, Vh = np.linalg.svd(X)
        diffs = - np.diff(S)
        noise_svals = list(range(n_start,n_end))
        mu = np.mean(diffs[list(range(n_start-1,n_end-1))])
        sigma = np.std(diffs[list(range(n_start-1,n_end-1))])
        num_of_sds = (diffs-mu)/sigma
        k = np.where(num_of_sds > thresh)
        return k



    def ProcPath(self):
        infolder = self.infolder
        outfolder = self.outfolder
        
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        out_prefix = outfolder
        if out_prefix[-1]!='/':
            out_prefix = out_prefix + r'/'
            
        in_prefix = infolder
        if in_prefix[-1]!='/':
            in_prefix = in_prefix + r'/'
        print('----------------------------------------------------------------' + '\n')
        print('-------- We are now processing the i/o paths for ---------------' + '\n')
        print(infolder)
        print('\n')
        
        print('----------------------------------------------------------------' + '\n')
        print('-------- We are saving the results at ---------------' + '\n')
        print(outfolder)
        print('\n')

        self.fn_a = in_prefix + r"csim.csv"
        self.fn_x = in_prefix + r"cellbygene_lognorm.csv"
        self.fn_l = in_prefix + r"gsim.csv"
        self.fn_w = out_prefix + r"banmfs_W.csv"
        self.fn_h = out_prefix + r"banmfs_H.csv"
        self.fn_x_out = out_prefix + r"banmfs_Ximp.csv"
        self.f_log = out_prefix +  r"banmfs_log.txt"
        self.f_err = out_prefix + r"banmfs_err.csv"

 
        print('Read the expression matrix from...')
        print(self.fn_x)
        print('\n')

        print('Read the cell similarity matrix from...')
        print(self.fn_a)
        print('\n')

        print('Read the gene similarity matrix from...')
        print(self.fn_l)
        print('\n')
        
        print('Save the W matrix at...')
        print(self.fn_w)
        print('\n')

        print('Save the H matrix at...')    
        print(self.fn_h)
        print('\n')

        print('Save the recovered matrix X at...')    
        print(self.fn_x_out)
        print('\n')

        print('Save the optimization records at...')    
        print(self.f_log)
        print('\n')

        print('----------- I/O preprocessings are finished! -------------------' + '\n')  
        print('----------------------------------------------------------------' + '\n')




    def ComputeDimension(self):
        """Compute the dimensions of matrices
        """
        for b in range(self.B):
            self.num_node += self.A[b][0].shape[0]
            self.num_attr += self.X[0][b].shape[1]

    def Update_t(self, Aij, Lrs, Xb, Mb, Wb, Hb, inds, child_pipe):
        """Xb , Hb and Wb are lists of matrices
           Aij and Lrs are matrices
        """

        if (inds[0] == inds[1]) & (inds[2] == inds[3]):
            Xir = Xb[0]
            Xjr = Xb[0]
            Xis = Xb[0]
            Xjs = Xb[0]
            Mir = Mb[0]
            Mjr = Mb[0]
            Mis = Mb[0]
            Mjs = Mb[0]
            Wi = Wb[0]
            Wj = Wb[0]
            Hr = Hb[0]
            Hs = Hb[0]
        elif (inds[0] == inds[1]) & (inds[2] != inds[3]):
            Xir, Xis = Xb
            Xjr, Xjs = Xb
            Mir, Mis = Mb
            Mjr, Mjs = Mb
            Wi = Wb[0]
            Wj = Wb[0]
            Hr, Hs = Hb
        elif (inds[0] != inds[1]) & (inds[2] == inds[3]):
            Xir, Xjr = Xb
            Xis, Xjs = Xb
            Mir, Mjr = Mb
            Mis, Mjs = Mb
            Wi, Wj = Wb
            Hr = Hb[0]
            Hs = Hb[0]
        else:
            Xir, Xjr, Xis, Xjs = Xb
            Mir, Mjr, Mis, Mjs = Mb
            Wi, Wj = Wb
            Hr, Hs = Hb

        # Gradient w.r.t. W^i and/or W^j
        if inds[0] != inds[1]:
            Wi_d = -0.5 * np.multiply(Mir, (Xir-Wi.dot(Hr))).dot(Hr.T) - 0.5 * np.multiply(Mis, (Xis-Wi.dot(Hs))).dot(Hs.T) -\
                    2 * self.gamma1 * (Aij-Wi.dot(Wj.T)).dot(Wj) + (self.alpha1 / self.B) * Wi
            Wj_d = -0.5 * np.multiply(Mjr, (Xjr - Wj.dot(Hr))).dot(Hr.T) - 0.5 * np.multiply(Mjs, (Xjs - Wj.dot(Hs))).dot(Hs.T)-\
                    2 * self.gamma1 * (Aij.T - Wj.dot(Wi.T)).dot(Wi) + (self.alpha1 / self.B) * Wj

        else:
            # Diagonal Aij
            #  if self.sparse_flag_x == 0:
            Wi_d = - np.multiply(Mir, (Xir-Wi.dot(Hr))).dot(Hr.T) - np.multiply(Mis, (Xis-Wi.dot(Hs))).dot(Hs.T) -\
                    4 * self.gamma1 * (Aij-Wi.dot(Wj.T)).dot(Wj) + 2 * (self.alpha1 / self.B) * Wi


        # Gradient w.r.t. H^r and H^s
        if inds[2] != inds[3]:
            Hr_d = -0.5 * (Wi.T).dot(np.multiply(Mir, (Xir-Wi.dot(Hr)))) - 0.5 * (Wj.T).dot(np.multiply(Mjr, (Xjr-Wj.dot(Hr)))) +\
                   self.gamma2 * Hs.dot(Lrs.T) + (self.alpha2 / self.B) * Hr
            Hs_d = -0.5 * (Wi.T).dot(np.multiply(Mis, (Xis-Wi.dot(Hs)))) - 0.5 * (Wj.T).dot(np.multiply(Mjs, (Xjs-Wj.dot(Hs)))) +\
                   self.gamma2 * Hr.dot(Lrs) + (self.alpha2 / self.B) * Hs

        else:
            Hr_d = - (Wi.T).dot(np.multiply(Mir, (Xir - Wi.dot(Hr)))) - Wj.T.dot(np.multiply(Mjr, (Xjr - Wj.dot(Hr)))) + \
                   2 * self.gamma2 * Hs.dot(Lrs.T) + 2 * (self.alpha2 / self.B) * Hr


        # Put gradients into the output pipe
        if (inds[0] != inds[1]) & (inds[2] != inds[3]):
            child_pipe.send([Wi_d, Wj_d, Hr_d, Hs_d])
        elif (inds[0] == inds[1]) & (inds[2] != inds[3]):
            child_pipe.send([Wi_d, Hr_d, Hs_d])
        elif (inds[0] != inds[1]) & (inds[2] == inds[3]):
            child_pipe.send([Wi_d, Wj_d, Hr_d])
        else:
            child_pipe.send([Wi_d, Hr_d])


    def BANMFS_t(self):
        """cost_function = ||M cdot (X - WH)||^2_f + gamma1 ||A - WW'||^2_f + gamma2 Tr(HLH') +
                           + alpha1 ||W||^2_f + alpha2 //H//^2_f
        """

        ## write to log
        f = open(self.f_log, 'w')
        f.write("NEW RUN\n" + self.fn_a + '\n' + self.fn_x + '\n' + "R: " + str(self.R) + ", B: " + str(self.B) + '\n')
        f.write("epsilon_start: " + str(self.epsilon_start) + '\n' + "epsilon_end: " + str(self.epsilon_end) + '\n')
        f.close()


        epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.max_iter
        B = self.B
        seed =self.seed


        W, H= Mat2Block.WHinit3(self.fn_x, self.B, self.k, self.shuf_flag, self.shufa, self.shufl, self.R, seed=seed)

        if seed == None:
            np.random.seed(11111)
            random.seed(11112)
        else:
            np.random.seed(seed)
            random.seed(seed+1)
        
        
        # BANMF-S processes
        self.time_start = time.time()  # start here!

        # check starting error
        print("Time Start")
        self.CheckError(B, W, H, self.time_start, self.time_start)

        epsilon = self.epsilon_start

        for t in range(self.max_iter):
            triplets = []
            ind_W = range(B)
            ind_H = range(B)
            while (len(ind_W)!= 0) & (len(ind_H) != 0):
                i = random.sample(ind_W, 1)[0]
                j = random.sample(ind_W, 1)[0]
                r = random.sample(ind_H, 1)[0]
                s = random.sample(ind_H, 1)[0]
                triplets.append([i, j, r, s])
                ind_W = [w for w in ind_W if w not in [i, j]]
                ind_H = [h for h in ind_H if h not in [r, s]]

            ## initialize processes
            procs = [[] for b in range(len(triplets))]
            self.pipes = [mp.Pipe() for b in range(self.B)]
            ## allocate jobs
            for b in range(len(triplets)):
                tplt = triplets[b]
                Ab = self.A[tplt[0]][tplt[1]]
                Lb = self.L[tplt[2]][tplt[3]]
                if (tplt[0] == tplt[1]) & (tplt[2] == tplt[3]):
                    Wb = [W[tplt[0]]]
                    Hb = [H[tplt[2]]]
                    Xb = [self.X[tplt[0]][tplt[2]]]
                    Mb = [self.M[tplt[0]][tplt[2]]]
                elif (tplt[0] == tplt[1]) & (tplt[2] != tplt[3]):
                    Wb = [W[tplt[0]]]
                    Hb = [H[tplt[2]], H[tplt[3]]]
                    Xb = [self.X[tplt[0]][tplt[2]], self.X[tplt[0]][tplt[3]]]
                    Mb = [self.M[tplt[0]][tplt[2]], self.M[tplt[0]][tplt[3]]]
                elif (tplt[0] != tplt[1]) & (tplt[2] == tplt[3]):
                    Wb = [W[tplt[0]], W[tplt[1]]]
                    Hb = [H[tplt[2]]]
                    Xb = [self.X[tplt[0]][tplt[2]], self.X[tplt[1]][tplt[2]]]
                    Mb = [self.M[tplt[0]][tplt[2]], self.M[tplt[1]][tplt[2]]]
                else:
                    Wb = [W[tplt[0]], W[tplt[1]]]
                    Hb = [H[tplt[2]], H[tplt[3]]]
                    Xb = [self.X[tplt[0]][tplt[2]], self.X[tplt[1]][tplt[2]],
                          self.X[tplt[0]][tplt[3]], self.X[tplt[1]][tplt[3]]]
                    Mb = [self.M[tplt[0]][tplt[2]], self.M[tplt[1]][tplt[2]],
                          self.M[tplt[0]][tplt[3]], self.M[tplt[1]][tplt[3]]]


                procs[b] = mp.Process(target=self.Update_t,
                                      args=(Ab, Lb, Xb, Mb, Wb, Hb, tplt, self.pipes[b][1]))

            # start
            for p in procs:
                p.start()

            b = 0
            for p in procs:
                tplt = triplets[b]
                wb_hb = self.pipes[b][0].recv()
                b += 1

                i, j, r, s = tplt
                if (i != j) & (r != s):
                  W[i] -= epsilon * wb_hb[0]
                  W[j] -= epsilon * wb_hb[1]
                  H[r] -= epsilon * wb_hb[2]
                  H[s] -= epsilon * wb_hb[3]
                elif (i != j) & (r == s):
                  W[i] -= epsilon * wb_hb[0]
                  W[j] -= epsilon * wb_hb[1]
                  H[r] -= epsilon * wb_hb[2]
                elif (i == j) & (r != s):
                    W[i] -= epsilon * wb_hb[0]
                    H[r] -= epsilon * wb_hb[1]
                    H[s] -= epsilon * wb_hb[2]
                else:
                    W[i] -= epsilon * wb_hb[0]
                    H[r] -= epsilon * wb_hb[1]
                p.join()

            # l1, all positive
            for b in range(self.B):
                continue

            for b in range(self.B):
                W[b] = W[b].clip(min=0)
                H[b] = H[b].clip(min=0)
            
            print("Iteration: ", t)
            time_inter = time.time()
            print("Time elapsed: ", time_inter - self.time_start)
            self.CheckError(B, W, H, self.time_start, time_inter)

            epsilon -= epsilon_decay

        self.time_end = time.time()
        print("Time cost: ", self.time_end - self.time_start)

        self.WriteTime2Log()
        pd.DataFrame(self.err_vec).to_csv(self.f_err)
        
        # turn bins into a integral matrix
        self.W = np.vstack(W)
        self.H = np.hstack(H)


    def CheckError(self, B, W, H, time_start, time_inter):
        err_A = 0.
        err_X = 0.
        err_L = 0.
        energy_W = 0.
        energy_H = 0.
        for i in range(B):
            energy_W += np.linalg.norm(W[i]) ** 2
            energy_H += np.linalg.norm(H[i]) ** 2
            for j in range(B):
                err_A += np.linalg.norm(self.A[i][j] - W[i].dot(W[j].T)) ** 2
                err_X += np.linalg.norm(np.multiply(self.M[i][j], self.X[i][j] - W[i].dot(H[j]))) ** 2
                err_L += np.trace(H[i].dot(self.L[i][j].dot(H[j].T)))

        err = err_X + self.gamma1 * err_A + self.gamma2 * err_L
        err = err ** 0.5

        self.err_t = err
        self.err_t_a = err_A ** 0.5
        self.err_t_x = err_X ** 0.5
        self.err_t_l = err_L ** 0.5
        self.err_t_w = energy_W ** 0.5
        self.err_t_h = energy_H ** 0.5
        print("Error: ", err)
        print("Error_A: ", err_A ** 0.5)
        print("Error_X: ", err_X ** 0.5)
        print("Error_L: ", err_L ** 0.5)
        print("Error_W: ", energy_W ** 0.5)
        print("Error_H: ", energy_H ** 0.5)

        self.err_vec.append(err)

        f = open(self.f_log, 'a')
        f.write('-----------------------------------------------------------------' + '\n')
        f.write("Time elapsed: " + str(time_inter - time_start) + '\n' + "Error: " + str(err) + '\n')
        f.write("Error_A: " + str(err_A ** 0.5) + '\n' + "Error_X: " + str(err_X ** 0.5) +
                '\n' + "Error_L: " + str(err_L ** 0.5) + '\n')
        f.write("Error_W: " + str(energy_W ** 0.5) + '\n' + "Error_H: " + str(energy_H ** 0.5) + '\n')
        f.write('-----------------------------------------------------------------' + '\n')
        f.close()


    def WriteTime2Log(self):
        f = open(self.f_log, 'a')
        f.write('-----------------------------Error Records----------------------' + '\n')
        f.write('Final Error Traces: ' + str(self.err_vec) + '\n')
        f.write('-----------------------------Final Records----------------------' + '\n')
        f.write('Time in Total: ' + str(self.time_end-self.time_start) + '\n')
        f.write('Final Error in Total: ' + str(self.err_t) + '\n')
        f.write('Final Error of Cell Similarity Matrix: ' + str(self.err_t_a) + '\n')
        f.write('Final Error of Gene Similarity Matrix: ' + str(self.err_t_l) + '\n')
        f.write('Final Error of X: ' + str(self.err_t_x) + '\n')
        f.close()

    def SaveWH(self):
        pd.DataFrame(self.W).to_csv(self.fn_w)
        pd.DataFrame(self.H).to_csv(self.fn_h)

    def SaveResults(self):
        pd.DataFrame(self.W).to_csv(self.fn_w)
        pd.DataFrame(self.H).to_csv(self.fn_h)
        self.X = np.matmul(self.W, self.H)
        pd.DataFrame(self.X).to_csv(self.fn_x_out)


