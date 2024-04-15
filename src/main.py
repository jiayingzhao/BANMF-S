from BANMFS import BANMFS
import argparse


if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("-i", "--infolder",help="Path of the input file.",type=str)
    parser.add_argument("-o", "--outfolder",help="Path of the output file.",type=str)
    parser.add_argument("--k", type=int, default=5, help="Number of clusters for kmeans initialization.")
    parser.add_argument("--R", type=int, default=15, help="Number of blocks for row. Default: 15")
    parser.add_argument("--B", type=int, default=15, help="Number of blocks for row. Default: 15")
    parser.add_argument("--schemeflag",type=str, default='b',help="Scheme Flag. Default: b")
    parser.add_argument("--alpha1", type=float, default=0.5, help="Alpha1. Default: 0.5")
    parser.add_argument("--alpha2", type=float, default=0.1, help="Alpha2. Default: 0.1")
    parser.add_argument("--gamma1", type=float, default=0.5, help="Gamma1. Default: 0.5")
    parser.add_argument("--gamma2", type=float, default=0.1, help="Gamma2. Default: 0.1")
    parser.add_argument("--epsilonstart", type=float, default=0.0001, help="Epsilon Start. Default: 0.0001")
    parser.add_argument("--epsilonend", type=float, default=0.00005, help="Epsilon End. Default: 0.00005")
    parser.add_argument("--maxiter", type=int, default=50, help="Max number of iterations. Default: 50")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility. Default: None")

    args = parser.parse_args()
    print(args)

    # vars()
    infolder = args.infolder
    outfolder = args.outfolder
    R = args.R
    B = args.B
    k = args.k
    scheme_flag = args.schemeflag
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    gamma1 = args.gamma1
    gamma2 = args.gamma2
    epsilon_start = args.epsilonstart
    epsilon_end = args.epsilonend
    max_iter = args.maxiter
    seed = args.seed

    banmfs = BANMFS(infolder, outfolder, R = R, B = B, k = k, scheme_flag = scheme_flag, alpha1 = alpha1, 
                 alpha2 = alpha2, gamma1 = gamma1, gamma2 = gamma2, 
                 epsilon_start = epsilon_start, epsilon_end = epsilon_end, 
                 max_iter = max_iter, seed = seed)
    
    banmfs.ReadInput()

    ## start iterations
    banmfs.BANMFS_t()

    # Save results
    banmfs.SaveResults()
















