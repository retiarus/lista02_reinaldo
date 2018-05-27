import scipy
import numpy as np

from scipy.optmize import minimize

def mlp(x, mean, variance, alpha):
    std = np.sqrt(variance)
    fvalue_x = (alpha/2.0) \
        * np.exp(alpha*mean+(alpha*alpha*variance)/2.0) \
        * np.power(x, -(alpha+1.0)) \
        * scipy.special.erfc((1.0/np.sqrt(2.0))
                             * (alpha*std-(np.ln(x)-mean)/std))
    return fvalue_x

def random_mlp(mean, variance, alpha, size):
    def aux():
        while(True):
            x = np.random.uniform()
            fvalue_x = mlp(x, mean, variance, alpha)
            if np.random.uniform() <= fvalue_x:
                yield x
            else:
                continue

    random_vector = np.fromiter(aux, float, size)

def find_alpha(moments):
    _mean = moments[0]
    _variance = moments[1]
    size = 2**16

    def objetive_function(alpha):
        mlp_vector = random_mlp(_mean, _variance, alpha, size)
        moments_mlp = calcMoments(mlp_vector)
        r = moments-moments_mlp

        return np.sqrt(r.dot(r))

    minimize(objetive_function)
