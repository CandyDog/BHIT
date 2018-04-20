import sys
import time
import numpy as np
import tensorflow as tf
from collections import OrderedDict

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
        print(">>> python tf_bhit.py input.txt output.txt 30000 29000 200 100 1 0.5")
        quit()
    
    dataset = np.loadtxt(sys.argv[1]).transpose()
    outFileName = sys.argv[2]
    iterNum = int(sys.argv[3])
    burninNum = int(sys.argv[4])
    obsNum = int(sys.argv[5])
    SNPNum = int(sys.argv[6])
    PhenoNum = int(sys.argv[7])
    MAF = float(sys.argv[8])
    
    return dataset, outFileName, iterNum, burninNum, obsNum, SNPNum, PhenoNum, MAF

def logLikeCont(contTensor):
    """
    Calculate logarithmic likelihood of only continuous variates given the 
    data tensor. Variates are assumed to be independent if there is only 
    one column, and from a multivariate Gaussian distribution if there are 
    multiple columns.

    Parameters
    ----------
    contTensor: Tensor
        Input tensor.
    
    Returns
    ----------
    logProb: Tensor
        Logarithmic likelihood of continuous variates.
    """
    varNum = tf.shape(contTensor)[0]
    obsNum = tf.cast(tf.shape(contTensor)[1], tf.float64)
    
    def func1():
        means = tf.reduce_mean(contTensor)
        sigma = tf.constant(1.0, dtype=tf.float64)
        
        nuVar = tf.multiply(NU0, tf.square(sigma))
        nuVar += tf.reduce_sum(tf.square(contTensor-means))
        nuVar += tf.multiply(KAPPA0, tf.multiply(
            obsNum/(KAPPA0+obsNum), tf.square(means-MU0)))

        res = (-1*tf.log(2*PI)*obsNum / 2 + tf.log(KAPPA0 / 
                (KAPPA0 + obsNum))/2 + tf.lgamma((NU0+obsNum)/2))
        res += (-1*tf.lgamma(NU0/2) + tf.log(NU0*tf.square(sigma)/
                2)*NU0/2 - tf.log(nuVar/2) * (NU0+obsNum)/2)
        return res
    
    def func2():
        means = tf.reduce_mean(contTensor, axis=1)
        lambda_arr = tf.diag(tf.ones_like(contTensor)[0])
        diff = tf.reshape(means-MU0, [1, -1])
        lambdaN = (lambda_arr + KAPPA0*obsNum/(KAPPA0+obsNum) * tf.matmul(
                    diff, diff, transpose_a=True, transpose_b=False))
#         lambdaN += (obsNum-1)*np.cov(contTensor)
 
        res = (-tf.log(PI) * obsNum * tf.cast(varNum, tf.float64) / 2 + tf.log(
            KAPPA0/(KAPPA0 + obsNum) * tf.cast(varNum, tf.float64) / 2))
        res += tf.log(tf.matrix_determinant(lambda_arr)) * NU0/2
        res -= tf.log(tf.matrix_determinant(lambdaN)) * (NU0+obsNum)/2
        res += tf.reduce_sum(tf.lgamma((NU0+obsNum)/2 - tf.range(varNum)/2) - 
                            tf.lgamma(NU0/2 - tf.range(varNum)/2))
        return res
    
    logProb = tf.case({tf.equal(varNum, 1): func1, tf.greater(varNum, 1): 
                       func2}, default=lambda:ZERO, exclusive=True)
        
    return logProb

def logLikeDisc(discTensor):
    """
    Calculate logarithmic likelihood of only discrete variates given the 
    data tensor. Variates are assumed to be independent if there is only 
    one column, and from a joint Dirichlet distribution if there are 
    multiple columns.

    Parameters
    ----------
    discTensor: Tensor
        Input tensor.
    
    Returns
    ----------
    logProb: Tensor
        Logarithmic likelihood of discrete variates.
    """
    logProb = tf.constant(0.0, dtype=tf.float64)
    combined_tensor = tf.reduce_join(discTensor, 0, separator=' ')
    unique_tensor, _, N = tf.unique_with_counts(combined_tensor)
    
    idx = tf.string_split(unique_tensor, delimiter=' ')
    idx = tf.sparse_tensor_to_dense(idx, default_value='1')
    idx = tf.string_to_number(idx, out_type=tf.int32) - 1
    
    alpha = tf.gather(Odds, idx)
    alpha = tf.reduce_prod(alpha, axis=1)
    n_plus_alpha = tf.add(alpha, tf.cast(N, alpha.dtype))
    
    logProb += tf.reduce_sum(tf.lgamma(n_plus_alpha) - tf.lgamma(alpha))
    logProb -= tf.lgamma(tf.reduce_sum(n_plus_alpha))
    return logProb
    
def logLikeDepe(discTensor, contTensor):
    """
    Calculate logarithmic likelihood of partitions with both continuous and 
    discrete variates by finding the continous rows corresponding to unique 
    discrete observations and calculating the probability those continous 
    observations came from a single multivariate Gaussian distribution.

    Parameters
    ----------
    discTensor: Tensor
        Input discrete tensor.
    contTensor: Tensor
        Input continous tensor.
    
    Returns
    ----------
    logProb: Tensor
        Logarithmic likelihood.
    """
    combined_tensor = tf.reduce_join(discTensor, axis=0, separator=' ')
    unique_tensor, _ = tf.unique(combined_tensor)
    
    def select_fn(elem):
        selected = tf.squeeze(tf.transpose(tf.gather(tf.transpose(contTensor), 
                    tf.where(tf.equal(combined_tensor, elem)))), [1])
        return logLikeCont(selected)
    
    logProb = tf.map_fn(lambda x: select_fn(x), unique_tensor, dtype=tf.float64)
    logProb = tf.reduce_sum(logProb)
    logProb += logLikeDisc(discTensor)
    
    return logProb

def calcProb(tensor1, tensor2):
    """
    Calculate likelihood given a dicrete tensor and continuous tensor.

    Parameters
    ----------
    tensor1: Tensor
        Input discrete tensor.

    tensor2: Tensor
        Input continuous tensor.

    Returns
    ----------
    res: Tensor
        Logarithmic likelihood.
    """
    shape1 = tf.shape(tensor1)
    shape2 = tf.shape(tensor2)
    
    pred_fn = OrderedDict([(tf.logical_and(tf.greater(shape1[0], 0), 
                tf.greater(shape2[0], 0)), lambda: logLikeDepe(tensor1, tensor2)),
                (tf.greater(shape1[0], 0), lambda: logLikeDisc(tensor1)),
                (tf.greater(shape2[0], 0), lambda: logLikeCont(tensor2))])
    res = tf.case(pred_fn, default=lambda: ZERO, exclusive=False)
    
    return res

# Initialize parameters used in the program.
(dataset, outFileName, iterNum, burninNum, obsNum, 
    SNPNum, PhenoNum, MAF) = init()
TotalNum = SNPNum + PhenoNum
Odds = np.array([(1-MAF)**2, 2*MAF*(1-MAF), MAF**2])

# Create TensorFlow graph.
graph = tf.Graph()
with graph.as_default():
    # Define TensorFlow constants.
    GeneData = tf.constant(dataset[:SNPNum].astype(np.int32).astype('str'))
    PhenoData = tf.constant(dataset[SNPNum:TotalNum])
    PI = tf.constant(np.pi, name='PI', dtype=tf.float64)
    ZERO = tf.zeros([], dtype=tf.float64)
    KAPPA0 = tf.constant(1.0, name='KAPPA', dtype=tf.float64)
    NU0 = tf.constant(PhenoNum+1, name='NU', dtype=tf.float64)
    MEANS = tf.reduce_mean(PhenoData, axis=1, name='MEANS')
    MU0 = tf.reduce_max(MEANS) + 2
    
    # Define TensorFlow placeholders.
    index1 = tf.placeholder(dtype=tf.int32)
    index2 = tf.placeholder(dtype=tf.int32)
    var1 = tf.placeholder(dtype=tf.int32)
    var2 = tf.placeholder(dtype=tf.int32)
    
    # Random number generator.
    u = tf.random_uniform([], dtype=tf.float64)
    
    Dx = index1[:SNPNum]
    Cx = index1[SNPNum:TotalNum]
    Dy = index2[:SNPNum]
    Cy = index2[SNPNum:TotalNum]
    
    Dxx = tf.squeeze(tf.gather(GeneData, tf.where(tf.equal(Dx, var1))), [1])
    Cxx = tf.squeeze(tf.gather(PhenoData, tf.where(tf.equal(Cx, var1))), [1])
    Dxy = tf.squeeze(tf.gather(GeneData, tf.where(tf.equal(Dx, var2))), [1])
    Cxy = tf.squeeze(tf.gather(PhenoData, tf.where(tf.equal(Cx, var2))), [1])
    Dyx = tf.squeeze(tf.gather(GeneData, tf.where(tf.equal(Dy, var1))), [1])
    Cyx = tf.squeeze(tf.gather(PhenoData, tf.where(tf.equal(Cy, var1))), [1])
    Dyy = tf.squeeze(tf.gather(GeneData, tf.where(tf.equal(Dy, var2))), [1])
    Cyy = tf.squeeze(tf.gather(PhenoData, tf.where(tf.equal(Cy, var2))), [1])
    
    pX = 0 
    pY = 0
    pX += calcProb(Dxx, Cxx)
    pX += calcProb(Dxy, Cxy)
    pY += calcProb(Dyx, Cyx)
    pY += calcProb(Dyy, Cyy)
     
    accept = tf.log(u) <= tf.minimum(ZERO, pY-pX)
    res = tf.cond(accept, lambda: index2, lambda: index1)

# Create TensorFlow session.
with tf.Session(graph=graph) as sess:
    start = time.time()
    # If you want to use TensorBoard to visualize graph, uncomment the following line.
    # writer = tf.summary.FileWriter('output/', sess.graph)
    sess.run(tf.global_variables_initializer())
    Ix = np.arange(TotalNum)
    iN = np.zeros([TotalNum, TotalNum+1])
    for i in range(iterNum):
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
            if (len(tmp1)!=1 or len(tmp2)!=1):
                break
        
        Ix = sess.run(res, {index1: Ix, index2: Iy, var1: x, var2: y})
            
        if (i+1) % 5000 == 0:
            print('Progress: %.2f%%' % ((i+1)/iterNum*100))
        
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
    print("Training Complete! Result:\n", mhResult)
    np.save(outFileName, mhResult)

end = time.time()
print("The whole program runs about %.2f s." % (end-start))