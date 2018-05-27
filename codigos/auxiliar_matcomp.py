''' Módulo auxiliar para a disicplina de computação aplicada '''

from numpy import correlate, mean, var, std, mod, trunc, zeros
from numpy import arange, array, linspace, min, max
from numpy import exp, log10, sqrt
from numpy import cumsum, ceil, power, int, array_split, polynomial, vstack

from scipy.stats import skew, kurtosis
from scipy.stats import norm, lognorm, beta
from scipy.stats import linregress
from scipy.optimize import leastsq, minimize

import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numdifftools as nd
import scipy.stats

def calcMoments(A):
    '''
    Calcula os quatro momentos da variável A
    input: A - numpy array, list, or tuple
    output: _moments - numpy array onde
            _moments[0] = valor médio de A
            _moments[1] = variância de A
            _moments[2] = assimetria de A
            _moments[3] = curtose de A
    '''

    _mean = mean(A)
    _var = var(A)
    _skew = skew(A)
    _kurtosis = kurtosis(A)

    _moments = array([_mean, _var, _skew, _kurtosis])

    return _moments


def fitting_normal_distribution(A):
    '''
    Fita uma distribuição normal (gaussiana) para os dados de entrada, e então
    plota o resultado, comparando a distribuição normal fitada, e a
    distribuição para os dados
    input: A - numpy array, list, or tuple
    '''

    # fitting normal distribution for varible A
    fitting_params = norm.fit(A)
    mu, sigma = fitting_params
    norm_dist_fitted = norm(*fitting_params)

    t = linspace(min(A), max(A), 100)
    norm_dist = norm(loc=mu, scale=sigma)

    # Plot normals
    f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))

    sns.distplot(A,
                 ax=ax,
                 bins=100,
                 norm_hist=True,
                 kde=False,
                 color='#afeeee',
                 label='Data A(t) ~ N(mu={0:.2f}, sigma={1:.2f})'.format(mu, sigma))

    ax.plot(t,
            norm_dist_fitted.pdf(t),
            lw=2,
            color='r',
            label='Modelo fitado A~N(mu={0:.2f}, sigma={1:.2f})'.format(norm_dist_fitted.mean(), norm_dist_fitted.std()))

    ax.plot(t,
            norm_dist.pdf(t),
            lw=2,
            color='g',
            ls=':',
            label='Modelo original A~N(mu={0:.2f}, sigma={1:.2f})'.format(norm_dist.mean(), norm_dist.std()))

    ax.legend(loc='upper right')

    # ax.title("Fitting de distribições de probabilidade para A")

    plt.show()


def fitting_lognormal_distribution(A):
    '''
    Fita uma distribuição lognormal para os dados de entrada, e então
    plota o resultado
    input: A - numpy array, list, or tuple
    '''

    # fitting lognormal distribution for varible A
    fitting_params = lognorm.fit(A)
    print("parametros de fitting: ", fitting_params)
    lognorm_dist_fitted = lognorm(*fitting_params)

    moments = lognorm_dist_fitted.stats('mvsk')
    print("        Fitado\t\t\t Original")
    print("mean : ", moments[0], '\t', mean(A))
    print("var  : ", moments[1], '\t', var(A))
    print("skew : ", moments[2], '\t', skew(A))
    print("kurt : ", moments[3], '\t', kurtosis(A))

    t = np.linspace(np.min(A), np.max(A), 100)

    # Plot lognormal
    f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))

    sns.distplot(A,
                 ax=ax,
                 bins=100,
                 norm_hist=True,
                 kde=False,
                 color='#afeeee',
                 label='Dado A: $mu$={0:.2f}, $sigma$={1:.2f})'.format(
                     A.mean(),
                     A.std()))

    ax.plot(t,
            lognorm_dist_fitted.pdf(t),
            lw=2,
            color='r',
            label='Lognormal fitting: $mu$={0:.2f}, $sigma$={1:.2f}'.format(
                lognorm_dist_fitted.mean(),
                lognorm_dist_fitted.std()))

    ax.legend(loc='upper right')

    plt.show()


def fitting_beta_distribution(A):
    '''
    Fita uma distribuição normal (gaussiana) para os dados de entrada, e então
    plota o resultado, comparando a distribuição normal fitada, e a
    distribuição para os dados
    input: A - numpy array, list, or tuple
    '''

    # fitting beta distribution for varible A
    scale = np.max(A) - np.min(A)
    fitting_params = beta.fit(A, floc=np.min(A)-0.0001,
                              fscale=scale+0.0002)
    print("parametros de fitting: ", fitting_params)
    a, b, _, _ = fitting_params
    beta_dist_fitted = beta(*fitting_params)

    moments = beta_dist_fitted.stats('mvsk')
    print("        Fitado\t\t\t Original")
    print("mean : ", moments[0], '\t', mean(A))
    print("var  : ", moments[1], '\t', var(A))
    print("skew : ", moments[2], '\t', skew(A))
    print("kurt : ", moments[3], '\t', kurtosis(A))

    t = np.linspace(np.min(A), np.max(A), 100)

    # Plot beta
    f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))

    sns.distplot(A,
                 ax=ax,
                 bins=100,
                 norm_hist=True,
                 kde=False,
                 color='#afeeee',
                 label='')

    ax.plot(t,
            beta_dist_fitted.pdf(t),
            lw=2,
            color='r',
            label='Modelo fitado A~beta(a={0:.2f}, b={1:.2f})'.format(a, b))

    ax.legend(loc='upper right')

    plt.show()


def fitting_lognormal_and_mlp_distribution(x):

    def mlp(m, alpha):
        '''
        MLP distribution function
        '''
        var = np.var(m)
        std = np.std(m)
        mu = np.mean(m)
        N = len(m)
        A = np.zeros(N)
        B = np.zeros(N)
        fx = np.zeros(N)

        for i in range(len(m)):
            A[i] = (alpha/2.0)*np.exp(alpha*mu+(alpha*alpha*var)/2.0)*np.power(np.exp(m[i]),-(alpha+1.0))
            B[i] = scipy.special.erfc((1.0/np.sqrt(2.0))*(alpha*std-(m[i]-mu)/std))
            fx[i] = A[i]*B[i]

        return fx

    mu = np.mean(x)
    sigma = np.sqrt(np.var(x))

    # The lognormal model fits to a variable whose log is normal
    # We create our variable whose log is normal 'exponenciating' the previous variable

    x_exp = np.exp(x)
    mu_exp = np.exp(mu)

    fitting_params_lognormal = lognorm.fit(x_exp, floc=0, scale=mu_exp)
    lognorm_dist_fitted = lognorm(*fitting_params_lognormal)
    t = np.linspace(min(x_exp), max(x_exp), 319)

    # Here is the magic I was looking for a long long time
    lognorm_dist = lognorm(s=sigma, loc=0, scale=np.exp(mu))

    # Plot lognormals
    f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))

    plt.xlabel(r'$exp(logM_{200})$')
    sns.distplot(x_exp,
                 ax=ax,
                 bins=100,
                 norm_hist=True,
                 kde=False,
                 label='Data exp(X)~N(mu={0:.2f}, sigma={1:.2f})\n X~LogNorm(mu={0:.2f}, sigma={1:.2f})'.format(mu, sigma))

    ax.plot(t,
            lognorm_dist.pdf(t),
            lw=2, color='r',
            label='Lognormal(mu={0:.2f}, sigma={1:.2f})'.format(lognorm_dist.mean(), lognorm_dist.std()))

    m = np.sort(x)

    ax.plot(np.exp(m),
            mlp(m, 15.0),
            lw=2,
            color='b',
            label='MLP')

    ax.legend(loc='upper right')
    plt.show()


def estimated_autocorrelation(A):
    ''' Calcula o vetor de autocorrelação para a entrada A
    input A:
    '''
    size = len(A)
    variance = var(A)
    A = A-mean(A)
    aux = correlate(A, A, mode='full')[-size:]
    return aux/(variance*(arange(size, 0, -1)))


def plot_estimated_autocorrelation(time, A, initial, final):
    plt.plot(time[initial:final], estimated_autocorrelation(A)[initial:final])
    plt.xlabel('time (s)')
    plt.ylabel('autocorrelation')
    plt.show()


def psd(data):
    """Calcula o PSD de uma série temporal."""

    # Define um intervalo para realizar o ajuste da reta
    INICIO = 10
    FIM = 800

    # O vetor com o tempo é o tamanho do número de pontos
    size = len(data)
    tempo = arange(len(data))

    # Define a frequência de amostragem
    dt = (tempo[-1] - tempo[0] / (size - 1))
    fs = 1 / dt

    # Calcula o PSD utilizando o MLAB
    power, freqs = mlab.psd(data, Fs=fs, NFFT=size, scale_by_freq=False)

    # Calcula a porcentagem de pontos utilizados na reta de ajuste
    totalFrequencias = len(freqs)
    totalPSD = FIM - INICIO
    porcentagemPSD = int(100 * totalPSD / totalFrequencias)

    # Seleciona os dados dentro do intervalo de seleção
    xdata = freqs[INICIO:FIM]
    ydata = power[INICIO:FIM]

    # Simula o erro
    yerr = 0.2 * ydata

    # Define uma função para calcular a Lei de Potência
    powerlaw = lambda x, amp, index: amp * (x**index)

    # Converte os dados para o formato LOG
    logx = log10(xdata)
    logy = log10(ydata)

    # Define a função para realizar o ajuste
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    logyerr = yerr / ydata

    # Calcula a reta de ajuste
    pinit = [1.0, -1.0]
    out = leastsq(errfunc,
                  pinit,
                  args=(logx, logy, logyerr), full_output=1)
    pfinal = out[0]
    covar = out[1]
    index = pfinal[1]
    amp = 10.0 ** pfinal[0]
    indexErr = sqrt(covar[0][0])
    ampErr = sqrt(covar[1][1]) * amp

    # Retorna os valores obtidos
    return freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM


def dfa1d(timeSeries, grau):
    """
    Calcula o DFA 1D (adaptado de Physionet), onde a escala cresce
    de acordo com a variável 'Boxratio'. Retorna o array 'vetoutput',
    onde a primeira coluna é o log da escala S e a segunda coluna é o
    log da função de flutuação.
    """

    # 1. A série temporal {Xk} com k = 1, ..., N
    # é integrada na chamada função perfil Y(k)
    x = mean(timeSeries)
    timeSeries = timeSeries - x
    yk = cumsum(timeSeries)
    tam = len(timeSeries)

    # 2. A série (ou perfil) Y(k) é dividida em N intervalos
    # não sobrepostos de tamanho S
    sf = ceil(tam / 4).astype(int)
    boxratio = power(2.0, 1.0 / 8.0)
    vetoutput = zeros(shape=(1, 2))

    s = 4
    while s <= sf:
        serie = yk
        if mod(tam, s) != 0:
            l = s*int(trunc(tam/s))
            serie = yk[0:l]

        t = arange(s, len(serie), s)
        v = array(array_split(serie, t))
        l = len(v)
        x = arange(1, s + 1)

        # 3. Calcula-se a variância para cada segmento v = 1,…, n_s:
        p = polynomial.polynomial.polyfit(x, v.T, grau)
        yfit = polynomial.polynomial.polyval(x, p)
        vetvar = var(v - yfit)

        # 4. Calcula-se a função de flutuação DFA como a média
        # das variâncias de cada intervalo
        fs = sqrt(mean(vetvar))
        vetoutput = vstack((vetoutput, [s, fs]))

        # A escala S cresce numa série geométrica
        s = ceil(s * boxratio).astype(int)

    # Array com o log da escala S e o log da função de flutuação
    vetoutput = log10(vetoutput[1::1, :])

    # Separa as colunas do vetor 'vetoutput'
    x = vetoutput[:, 0]
    y = vetoutput[:, 1]

    # Regressão linear
    slope, intercept, _, _, _ = linregress(x, y)

    # Calcula a reta de inclinação
    predict_y = intercept + slope * x

    # Calcula o erro
    pred_error = y - predict_y

    # Retorna o valor do ALFA, o vetor 'vetoutput', os vetores X e Y,
    # o vetor com os valores da reta de inclinação e o vetor de erros
    return slope, vetoutput, x, y, predict_y, pred_error


def plot_psd_dfa(data, tituloPrincipal=None):
    # Desabilita as mensagens de erro do Numpy (warnings)
    #old_settings = np.seterr(divide = 'ignore', invalid = 'ignore', over = 'ignore')

    N = 10
    print("Original time series data (%d points): \n" % (len(data)))
    print("First %d points: %s\n" % (N, data[0:10]))
    print()

    # -----------------------------------------------------------------
    # Parâmetros gerais de plotagem
    # -----------------------------------------------------------------

    # Define os subplots
    fig = plt.figure()
    fig.subplots_adjust(hspace=.3, wspace=.2)

    # Tamanho das fontes
    tamanhoFonteEixoX = 16
    tamanhoFonteEixoY = 16
    tamanhoFonteTitulo = 16
    tamanhoFontePrincipal = 25

    # -----------------------------------------------------------------
    # Plotagem da série original
    # -----------------------------------------------------------------

    # Define as cores da plotagem
    corSerieOriginal = 'r'

    # Título dos eixos da série original
    textoEixoX = 'Tempo'
    textoEixoY = 'Amplitude'
    textoTituloOriginal = 'Original Time Series Data'

    print("1. Plotting time series data...")

    # Plotagem da série de dados
    original = fig.add_subplot(2, 1, 1)
    original.plot(data, '-', color=corSerieOriginal)
    original.set_title(textoTituloOriginal, fontsize=tamanhoFonteTitulo)
    original.set_xlabel(textoEixoX, fontsize=tamanhoFonteEixoX)
    original.set_ylabel(textoEixoY, fontsize=tamanhoFonteEixoY)
    original.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    original.grid()

    # -----------------------------------------------------------------
    # Cálculo e plotagem do PSD
    # -----------------------------------------------------------------

    # Calcula o PSD
    freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = psd(data)
    if len(freqs) < FIM:
        FIM = len(freqs)-1

    # O valor do beta equivale ao index
    b = index

    # Define as cores da plotagem
    corPSD1 = 'k'
    corPSD2 = 'navy'

    # Título dos eixos do PSD
    textoPSDX = 'Frequência (Hz)'
    textoPSDY = 'Potência'
    textoTituloPSD = r'Power Spectrum Density $\beta$ = '

    print("2. Plotting Power Spectrum Density...")

    # Plotagem do PSD
    PSD = fig.add_subplot(2, 2, 3)
    PSD.plot(freqs, power, '-', color=corPSD1, alpha=0.7)
    PSD.plot(xdata, ydata, color=corPSD2, alpha=0.8)
    PSD.axvline(freqs[INICIO], color=corPSD2, linestyle='--')
    PSD.axvline(freqs[FIM], color=corPSD2, linestyle='--')
    PSD.plot(xdata,
             powerlaw(xdata, amp, index),
             'r-',
             linewidth=1.5,
             label='$%.4f$' % (b))
    PSD.set_xlabel(textoPSDX, fontsize=tamanhoFonteEixoX)
    PSD.set_ylabel(textoPSDY, fontsize=tamanhoFonteEixoY)
    PSD.set_title(textoTituloPSD + '%.4f' % (b),
                  loc='center',
                  fontsize=tamanhoFonteTitulo)
    PSD.set_yscale('log')
    PSD.set_xscale('log')
    PSD.grid()

    # -----------------------------------------------------------------
    # Cálculo e plotagem do DFA
    # -----------------------------------------------------------------

    # Calcula o DFA 1D
    alfa, vetoutput, x, y, reta, erro = dfa1d(data, 1)

    # Verifica se o DFA possui um valor válido
    # Em caso afirmativo, faz a plotagem
    if not math.isnan(alfa):

        # Define as cores da plotagem
        corDFA = 'darkmagenta'

        # Título dos eixos do DFA
        textoDFAX = '$log_{10}$ (s)'
        textoDFAY = '$log_{10}$ F(s)'
        textoTituloDFA = r'Detrended Fluctuation Analysis $\alpha$ = '

        print("3. Plotting Detrended Fluctuation Analysis...")

        # Plotagem do DFA
        DFA = fig.add_subplot(2, 2, 4)
        DFA.plot(x, y, 's',
                 color=corDFA,
                 markersize=4,
                 markeredgecolor='r',
                 markerfacecolor='None',
                 alpha=0.8)
        DFA.plot(x, reta, '-', color=corDFA, linewidth=1.5)
        DFA.set_title(textoTituloDFA + '%.4f' % (alfa),
                      loc='center',
                      fontsize=tamanhoFonteTitulo)
        DFA.set_xlabel(textoDFAX, fontsize=tamanhoFonteEixoX)
        DFA.set_ylabel(textoDFAY, fontsize=tamanhoFonteEixoY)
        DFA.grid()

    else:
        DFA = fig.add_subplot(2, 2, 4)
        DFA.set_title(textoTituloDFA + 'N.A.',
                      loc='center',
                      fontsiz=tamanhoFonteTitulo)
        DFA.grid()

        # -----------------------------------------------------------------
        # Exibe e salva a figura
        # -----------------------------------------------------------------
    if tituloPrincipal:
        plt.suptitle(tituloPrincipal, fontsize=tamanhoFontePrincipal)

    fig.set_size_inches(15, 9)
    plt.show()


class Lognoral_mlp():
    '''
    '''
    def __init__(self, mu=None, sigma=None, alpha=None):
        self._mu = mu
        self._sigma = sigma
        self._alpha = alpha

    def _moment(self, k, mu, sigma, alpha):
        if alpha > k:
            return ((alpha)/(alpha-k))*exp(mu*k+sigma**2.0*k**2.0/2)

    def _mean(self, mu, sigma, alpha):
        return self._moment(1, mu, sigma, alpha)

    def _variance(self, mu, sigma, alpha):
        if alpha > 2.0:
            return alpha*exp(sigma**2.0+2*mu) \
                * (exp(sigma**2.0)/(alpha-2.0)-alpha/(alpha-1)**2.0)

    def _std(self, mu, sigma, alpha):
        return np.sqrt(self._variance(mu, sigma, alpha))

    def _skew(self, mu, sigma, alpha):
        mean = self._mean(mu, sigma, alpha)
        std = self._std(mu, sigma, alpha)
        moment_3 = self._moment(3, mu, sigma, alpha)

        return (moment_3 - 3.0*mean*std**2.0 - mean**3.0)/(std**3.0)

    def _kurtosis(self, mu, sigma, alpha):
        mean = self._mean(mu, sigma, alpha)
        std = self._std(mu, sigma, alpha)
        moment_3 = self._moment(3, mu, sigma, alpha)
        moment_4 = self._moment(4, mu, sigma, alpha)

        std2 = std**2
        std4 = std**4

        return moment_4/std4 \
            + (mean/std4)*(3*mean**3 + 6*mean*std2 - 4*moment_3)

    def fit(self, A):
        A_mean = np.mean(A)
        A_variance = np.var(A)
        A_skew = scipy.stats.skew(A)
        A_kurtosis = scipy.stats.kurtosis(A)

        array_moments = array([A_mean,
                               A_variance,
                               A_skew,
                               A_kurtosis])

        def objetivo(f):
            mu = f[0]
            sigma = f[1]
            alpha = f[2]

            if alpha > 4:
                pass

            moments = array([self._mean(mu, sigma, alpha),
                            self._variance(mu, sigma, alpha),
                            self._skew(mu, sigma, alpha),
                            self._kurtosis(mu, sigma, alpha)])

            aux = array_moments-moments

            return aux.dot(aux)

        jac_objetivo = lambda f: nd.Jacobian(objetivo)(f).ravel()

        f0 = array([0.0, 1.0, 5.0])

        f = minimize(objetivo,
                     f0,
                     method='Newton-CG',
                     jac=jac_objetivo,
                     tol=1e-8)

        self._mu = f.x[0]
        self._sigma = f.x[1]
        self._alpha = f.x[2]

        return f

    def moment(self, k):
        return self._moment(k, self._mu, self._sigma, self._alpha)

    def mean(self):
        return self._mean(self._mu, self._sigma, self._alpha)

    def variance(self):
        return self._variance(self._mu, self._sigma, self._alpha)

    def std(self):
        return self._std(self._mu, self._sigma, self._alpha)

    def skew(self):
        return self._skew(self._mu, self._sigma, self._alpha)

    def kurtosis(self):
        return self._kurtosis(self._mu, self._sigma, self._alpha)

    def pdf(self, x):
        A = (self._alpha/2.0) \
            * exp(self._alpha*self._mu + (self._alpha**2.0 *
                                          self._sigma**2.0)/2.0) \
            * power(x, -(self._alpha + 1.0))

        B = scipy.special.erfc((1.0/np.sqrt(2.0))*(self._alpha*self._sigma-(np.log(x)-self._mu)/self._sigma))

        return A*B
