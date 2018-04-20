import sys
import time
import scipy.special
import numpy as np

#==============================================================================
# Helper functions defines here
#==============================================================================
def init():
    """
    Intialize parameters that will be used in the program.

    Parameters
    ----------
    None

    Returns
    ----------
    dataset: ndarray
        The whole dataset read from the input file.
    outFileName: String
        Name of output file.
    iterNum: int
        Number of iteration.
    burninNum: int
        Burn-in number of iterations.
    obsNum: int
        Number of observations, e.g., number of population.
    SNPNum: int
        Number of single nucleotide polymorphisms (SNPs).
    PhenoNum: int
        Number of phenotype types.
    MAF: float
        Minor Allele Frequency, should be less than 1.
    """
    if len(sys.argv) != 9:
        print("Number of arguments don't meet program's requirements.", 
              "Please Check again!")
        print("You should specify inputFileName, outputFileName, iterNum,", 
              "burninNum, obsNum, \n\tSNPNum, PhenoNum, MAF in the arguments.")
        print("Here is an example: ")
        print(">>> python np_bhit.py input.txt output.npy 30000 29000 200 100 1 0.5")
        quit()
    
    dataset = np.loadtxt(sys.argv[1])
    outFileName = sys.argv[2]
    iterNum = int(sys.argv[3])
    burninNum = int(sys.argv[4])
    obsNum = int(sys.argv[5])
    SNPNum = int(sys.argv[6])
    PhenoNum = int(sys.argv[7])
    MAF = float(sys.argv[8])
    
    return dataset, outFileName, iterNum, burninNum, obsNum, SNPNum, PhenoNum, MAF

def unique(arr, return_counts=True):
    """
    Find the unique elements of an array. Return the sorted unique elements 
    of an array, and the number of times each unique value comes up in the 
    input array if `return_counts` is True.

    Parameters
    ----------
    arr: array_like
        Input array.
    return_counts: bool
        If True, also return the number of times each unique item appears
        in `arr`.

    Returns
    ----------
    unique: ndarray
        The sorted unique values.
    unique_counts: ndarray
        The number of times each of the unique values comes up in the
        original array.
    """
    arr = np.asanyarray(arr)    
    orig_shape, orig_dtype = arr.shape, arr.dtype
    # Must reshape to a contiguous 2D array for this to work...
    arr = arr.reshape(orig_shape[0], -1)
    arr = np.ascontiguousarray(arr)

    if arr.dtype.char in (np.typecodes['AllInteger'] +
                         np.typecodes['Datetime'] + 'S'):
        # Optimization: Creating a view of your data with a np.void data type of
        # size the number of bytes in a full row. Handles any type where items
        # have a unique binary representation, i.e. 0 is only 0, not +0 and -0.
        dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), arr.dtype) for i in range(arr.shape[1])]

    try:
        consolidated = arr.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=arr.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, 0)
        return uniq

    tmp = np.asanyarray(consolidated).flatten()

    if tmp.size == 0:
        if not return_counts:
            output = tmp
        else:
            output = (tmp,)
            output += (np.empty(0, np.intp),)
    else:
        tmp.sort()
        aux = tmp
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))
        
        if not return_counts:
            output = aux[flag]
        else:
            output = (aux[flag],)
            idx = np.concatenate(np.nonzero(flag) + ([tmp.size],))
            output += (np.diff(idx),)
    
    if not return_counts:
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return (uniq,) + output[1:]

def logLikeCont(contArr):
    """
    Calculate logarithmic likelihood of only continuous variates given the 
    data array. Variates are assumed to be independent if there is only 
    one column, and from a multivariate Gaussian distribution if there are 
    multiple columns.

    Parameters
    ----------
    contArr: array_like
        Input array.
    
    Returns
    ----------
    logProb: float
        Logarithmic likelihood of continuous variates.
    """
    Y = np.asanyarray(contArr)
    obsNum, varNum = Y.shape
    logProb = 0
    
    if varNum == 0:
        pass
    elif varNum == 1:
        sigma = 1
        mean = np.average(Y)
        nuVar = NU0 * sigma**2
        nuVar += (obsNum-1) * np.var(Y)
        nuVar += KAPPA0*obsNum/(KAPPA0+obsNum) * (mean-MU0)**2

        logProb = -1*np.log(2*np.pi)*obsNum/2 + np.log(KAPPA0/(KAPPA0 + obsNum)) / 2 + scipy.special.gammaln((NU0+obsNum)/2)
        logProb += (-1*scipy.special.gammaln(NU0/2) + np.log(NU0*sigma**2/2)*NU0/2 - np.log(nuVar/2) * (NU0+obsNum)/2)
    
    # The below code was not fully tested.
    else:
        means = np.average(Y, axis=0)
        lambda_arr = np.diag([1]*varNum)
        diff = np.array(means - MU0)[:,None]
        lambdaN = (lambda_arr + KAPPA0 * obsNum / (KAPPA0+obsNum)
                    * diff.dot(diff.transpose()))
        lambdaN += (obsNum-1)*np.cov(Y, rowvar=False, bias=False)

        logProb = (-np.log(np.pi) * obsNum * varNum / 2 + np.log(KAPPA0/
            (KAPPA0 + obsNum) * varNum / 2))
        logProb += np.log(np.linalg.det(lambda_arr)) * NU0/2
        logProb -= np.log(np.linalg.det(lambdaN)) * (NU0+obsNum)/2
        logProb += np.sum(scipy.special.gammaln((NU0+obsNum)/2 - 
            np.arange(varNum)/2) - scipy.special.gammaln(NU0/2 - 
            np.arange(varNum)/2))
        
    return logProb

def logLikeDisc(discArr):
    """
    Calculate logarithmic likelihood of only discrete variates given the 
    data array. Variates are assumed to be independent if there is only 
    one column, and from a joint Dirichlet distribution if there are 
    multiple columns.

    Parameters
    ----------
    discArr: array_like
        Input array.
    
    Returns
    ----------
    logProb: float
        Logarithmic likelihood of discrete variates.
    """
    X = np.asanyarray(discArr)
    uniqueArr, N = unique(X)
    
    alpha = Odds[uniqueArr-1]
    alpha = np.prod(alpha, axis=1)
    n_plus_alpha = N + alpha
    
    logProb = np.sum(scipy.special.gammaln(n_plus_alpha) - scipy.special.gammaln(alpha))
    logProb -= scipy.special.gammaln(np.sum(n_plus_alpha))
    return logProb

def logLikeDepe(discArr, contArr):
    """
    Calculate logarithmic likelihood of partitions with both continuous and 
    discrete variates by finding the continous rows corresponding to unique 
    discrete observations and calculating the probability those continous 
    observations came from a single multivariate Gaussian distribution.

    Parameters
    ----------
    discArr: array_like
        Input discrete array.
    contArr: array_like
        Input continous array.
    
    Returns
    ----------
    logProb: float
        Logarithmic likelihood.
    """
    X = np.asanyarray(discArr)
    Y = np.asanyarray(contArr)
    variations = unique(X, return_counts=False)
    logProb = 0
    
    for v in variations:
        corres_row = np.prod((X==v), axis=1)
        corres_Y = Y[corres_row==1]
        logProb += logLikeCont(corres_Y)
    logProb += logLikeDisc(X)

    return logProb

def metroHast(iterNum, burninNum):
    """
    The Metropolis–Hastings algorithm is a Markov chain Monte Carlo (MCMC) 
    method for obtaining a sequence of random samples from a probability 
    distribution for which direct sampling is difficult. The reult returned 
    is the final probability matrix for each covariate.
    
    Parameters
    ----------
    iterNum: number
        Number of iteration of MCMC.
    burninNum: number
        Number of burn-in of MCMC.

    Returns
    ----------
    mhResult: ndarray
        Final partition matrix for each covariate.
    Ix: ndarray
        Final index vector.
    """
    Ix = np.arange(TotalNum)
    Dx = Ix[:SNPNum]
    Cx = Ix[SNPNum:TotalNum]
    iN = np.zeros([TotalNum, TotalNum+1])
    
    # Uncomment if you want to trace probabilities.
    # trace = np.zeros(iterNum)
    # trace[0] += np.sum([logLikDisc(Genotype[:, Dx==col]) for col in range(SNPNum)])
    # trace[0] += np.sum([logLikCont(Phenotype[:, Cx==col]) for col in range(SNPNum,TotalNum)])

    # Main Metropolis-Hastings loop.
    for i in range(1, iterNum):
        # Select an index, then change it to another index randomly.
        while True:
            # Sort the number to ensure changing from small index to big one.
            x, y = np.sort(np.random.choice(Ix, 2, False))
            k = np.where(Ix == x)[0]
            
            if len(k) > 1:
                k = np.random.choice(k, 1)
          
            Iy = np.array(Ix)
            Iy[k] = y
          
            tmp1 = np.where(Ix == x)[0]
            tmp2 = np.where(Iy == y)[0]
            if (len(tmp1)>1 or len(tmp2)>1):
                break

        # Create the proposed indicator vector.
        Dy = Iy[:SNPNum]
        Cy = Iy[SNPNum:TotalNum]
        Cxx = Phenotype[:,Cx == x]
        Cxy = Phenotype[:,Cx == y]
        Cyx = Phenotype[:,Cy == x]
        Cyy = Phenotype[:,Cy == y]
        Dxx = Genotype[:,Dx == x]
        Dxy = Genotype[:,Dx == y]
        Dyx = Genotype[:,Dy == x]
        Dyy = Genotype[:,Dy == y]
        
        # Calculate log likelihoods.
        old_prob = 0
        new_prob = 0
        
        # Likelihood of current partition x.
        if Cxx.size != 0:
            if Dxx.size != 0:
                old_prob += logLikeDepe(Dxx, Cxx)
            else:
                old_prob += logLikeCont(Cxx)
        elif Dxx.size != 0:
            old_prob += logLikeDisc(Dxx)
        
        # Likelihood of current partition x.
        if Cxy.size != 0:
            if Dxy.size != 0:
                old_prob += logLikeDepe(Dxy, Cxy)
            else:
                old_prob += logLikeCont(Cxy)
        elif Dxy.size != 0:
            old_prob += logLikeDisc(Dxy)
        
        # Likelihood of proposed partition y.
        if Cyx.size != 0:
            if Dyx.size != 0:
                new_prob += logLikeDepe(Dyx, Cyx)
            else:
                new_prob += logLikeCont(Cyx)
        elif Dyx.size != 0:
            new_prob += logLikeDisc(Dyx)

        # Likelihood of proposed partition y.
        if Cyy.size != 0:
            if Dyy.size != 0:
                new_prob += logLikeDepe(Dyy, Cyy)
            else:
                new_prob += logLikeCont(Cyy)
        elif Dyy.size != 0:
            new_prob += logLikeDisc(Dyy)
        
        # Uncomment if you want to trace probabilities.
        # trace[i] = trace[i-1]
        
        # Check if proposal is accepted, if so, update everything.
        accept = np.log(np.random.rand()) <= min(0, new_prob-old_prob)
        if accept:
            Ix = np.array(Iy)
            Cx = np.array(Cy)
            Dx = np.array(Dy)
            # trace[i] += new_prob-old_prob
        
        if (i+1) % 5000 == 0:
            print("Progress: %.2f%%" % ((i+1)/iterNum*100))
            
        # When MCMC gets convergence, we start to count indices.
        if i >= burninNum:
            for h in range(TotalNum):
                tmp = np.where(Ix==h)[0]
                if (len(tmp) == 1):
                    iN[tmp, 0] += 1
                elif (len(tmp) > 1):
                    iN[tmp, h+1] += 1

    # Normalize rows of result to show proportions.
    iN = iN/np.sum(iN, axis=1)[:,None]
    cst = np.sum(iN, axis=0)[1:]

    # Remove columns that have no values in them.
    tmp = np.where(cst != 0)[0]
    iNt = iN[:, 1:]
    if len(tmp) == 0:
        iNtr = iNt[:,1]
    else:
        iNtr = iNt[:,tmp]
    
    # Have the first column correspond to all independent variates, 
    # and others are partitions with more than one variate.
    mhResult = np.zeros([TotalNum+1, len(tmp)+2])
    mhResult[0, 1:] = np.arange(len(tmp)+1)
    mhResult[1:SNPNum+1, 0] = np.arange(1, SNPNum+1)
    mhResult[SNPNum+1:, 0] = np.arange(1, PhenoNum+1)
    mhResult[1:,1] = iN[:TotalNum,0]
    mhResult[1:,2:] = iNtr
    
    # Uncomment if you want to save trace file.
    # np.savetxt(outFileName+"_trace", trace, fmt='%.1f')
    
    return mhResult, Ix

#==============================================================================
# Main entry begins here
#==============================================================================
start = time.time()

# Initialize parameters for later usage.
(dataset, outFileName, iterNum, burninNum, obsNum, 
    SNPNum, PhenoNum, MAF) = init()
TotalNum = SNPNum + PhenoNum
Genotype = dataset[:, :SNPNum].astype(int)
Phenotype = dataset[:, SNPNum:TotalNum]
Odds = np.array([(1-MAF)**2, 2*MAF*(1-MAF), MAF**2])

# Define hyper-parameters here.
KAPPA0 = 1
NU0 = PhenoNum + 1
MEANS = np.average(Phenotype, axis=0)
MU0 = max(MEANS) + 2

# Check if input file format is valid.
if (Genotype.shape[0] != Phenotype.shape[0]):
    print("Discrete and continuous data must have same number of rows!\n")
    quit()
print("Initialization Completed!")

# Detection using Metropolis–Hastings algorithm.
mhResult, index = metroHast(iterNum, burninNum)
print(index)        # for debugging usage, can be removed
np.savetxt(outFileName, mhResult)       # save output file

# Output running time of the program.
end = time.time()
print("The whole program runs about %.2fs." % (end-start))
