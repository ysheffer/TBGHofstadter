# from scipy.special import eval_genlaguerre
# import numba_scipy
# from math import gamma
# from functools import lru_cache
# import time
# import scipy.special as sc
# import scipy.special.cython_special
import numpy as np
from tqdm import tqdm
from scipy import linalg
import matplotlib.pyplot as plt
from numba import njit
from numba.extending import get_cython_function_address
import ctypes
from tqdm import tqdm
import warnings
import datetime
import time
import dataclasses
import os.path

###Project folder###
path = "/home/yarden/Dropbox/MSc project/MagneticTBG/"

### General constants ###
pi = np.pi
m_s_angstrom_to_mev = 0.006582  # m/s/angstrom*hbar to mev
hbar_e_angstrom2_to_tesla = 65821.2  # hbar/e/angstrom^2 in Teslas
pauli = np.array([[[1, 0], [0, 1]],
                  [[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]
                  ])


# rotation matrix in 2D
@njit
def rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


### Define cfuncs for evaluation of genlaguerre and gamma, for use in fnm ###
_dble = ctypes.c_double
_long = ctypes.c_long
addr = get_cython_function_address('scipy.special.cython_special', '__pyx_fuse_1_1eval_genlaguerre')
_c_eval_genlaguerre_functype = ctypes.CFUNCTYPE(_dble, _long, _dble, _dble)
_c_eval_genlaguerre = _c_eval_genlaguerre_functype(addr)

addr = get_cython_function_address('scipy.special.cython_special', 'gammaln')
_c_gammaln_functype = ctypes.CFUNCTYPE(_dble, _dble)
_c_gammaln = _c_gammaln_functype(addr)


@njit
def gamma_rat(m, n):
    """ Returns sqrt(gamma(m+1)/gamma(n+1)) """
    return np.exp((_c_gammaln(m + 1) - _c_gammaln(n + 1)) / 2)


@njit
def fnm(zx, zy, n, m):
    """ Results of gaussian integrals for evaluation of interaction between landau levels. Evaluated the function F_nm
    defined in PhysRevB.84.035440.
    Jitted to improve performence.
    """
    z2 = zx ** 2 + zy ** 2
    if n >= m:
        return np.exp((_c_gammaln(m + 1) - _c_gammaln(n + 1)) / 2 +
                      np.log(-zx + 1j * zy) * (n - m) -
                      z2 / 2) * _c_eval_genlaguerre(m, n - m, z2)
    else:
        return np.exp((_c_gammaln(n + 1) - _c_gammaln(m + 1)) / 2 +
                      np.log(zx + 1j * zy) * (m - n) -
                      z2 / 2) * _c_eval_genlaguerre(n, m - n, z2)
        # np.conj(fnm(-zx, -zy, m, n))


def gen_pqlist(max_q, min_alpha=0, max_alpha=1):
    """
    Generates a list of rational numbers between min_alpha and max_alpha.

    :param max_q: maximal denominator
    :param min_alpha: minimal p/q
    :param max_alpha: maximal p/q
    :return: plist, qlist, pqlist: arrays with the list of p's, q's and p/q's.
    """
    plist = np.array([])
    qlist = np.array([])
    for q in range(1, max_q + 1):
        for p in range(int(max(np.ceil(q * min_alpha), 1)), int(np.floor(q * max_alpha)) + 1):
            if np.gcd(p, q) == 1:
                plist = np.append(plist, p)
                qlist = np.append(qlist, q)
    pqlist = plist / qlist
    s_ind = np.argsort(pqlist)
    plist = plist[s_ind]
    qlist = qlist[s_ind]
    pqlist = pqlist[s_ind]
    return plist, qlist, pqlist


@njit  # (cache=True, parallel=True)
def _hamiltonian_aux(kx, ky, p: int, q: int, max_l: int, theta: float, ktheta: float, vel: float, w1: float, w2: float,
                     sublat_pot: float, layer_pot: float):
    """
    Hamiltonian of the Kronker product (p guiding)*(2 layer)*(n level)*(2 sublattice).
    Construction follows Bistrizer & Macdonald PhysRevB.84.035440

    This is a numba-jitted auxillery function, do not use.
    """

    # if max_l >= 170:
    #     raise ValueError("max_l>170 is not supported")

    p = int(p)
    q = int(q)
    h = np.zeros((p * 2 * (max_l + 1) * 2, p * 2 * (max_l + 1) * 2), dtype=np.complex128)

    # magnetic parameters
    b = np.sqrt(3) / 8 / pi * ktheta ** 2 * p / q
    mag_l = b ** (-1 / 2)
    wc = np.sqrt(2) * vel / mag_l * m_s_angstrom_to_mev

    # scattering parameters
    phi = 2 * pi / 3
    exphi = np.exp(1j * phi)
    T1 = np.array([[w1, w2], [w2, w1]], dtype=np.complex128)
    T2 = np.array([[w1 / exphi, w2], [w2 * exphi, w1 / exphi]])
    T3 = np.array([[w1 * exphi, w2], [w2 / exphi, w1 * exphi]])
    q1 = ktheta * np.array([0, -1])
    q2 = ktheta * np.array([np.sqrt(3), 1]) / 2
    q3 = ktheta * np.array([-np.sqrt(3), 1]) / 2
    z1 = q1 * mag_l / np.sqrt(2)
    z2 = q2 * mag_l / np.sqrt(2)
    z3 = q3 * mag_l / np.sqrt(2)
    delta = 2 * pi * q / p / ktheta  # np.sqrt(3) * ktheta * mag_l ** 2 / 2
    y0 = kx * mag_l ** 2
    ind = np.reshape(np.arange(p * 2 * (max_l + 1) * 2), (p, 2, (max_l + 1), 2))

    # diagonal term
    for level in range(max_l):
        for y in range(p):
            for layer in range(2):
                # landau level energies
                h[ind[y, layer, level, 1], ind[y, layer, level + 1, 0]] = -wc * np.exp(
                    1j * ((-1) ** layer) * theta / 2) * np.sqrt(level + 1.)

                # sublattice and layer potential
                for sublat in range(2):
                    ii = ind[y, layer, level, sublat]
                    h[ii, ii] = (-1) ** sublat * sublat_pot + (-1) ** layer * layer_pot

    for y in range(p):
        for layer in range(2):
            h[ind[y, layer, max_l, 1], ind[y, layer, max_l, 1]] = 1e4

    # off-diagonal term
    # m,n are level indices
    # alpha,beta are sublattice indices
    # j is the guiding center coordinate
    for m in range(max_l + 1):
        for n in range(max_l + 1):
            for beta in range(2):
                for alpha in range(2):
                    for j in range(p):
                        # top scattering
                        h[ind[j, 0, m, beta], ind[j, 1, n, alpha]] += \
                            T1[alpha, beta] * fnm(z1[0], z1[1], n, m) * np.exp(
                                -1j * ktheta * y0 - 4j * pi * q * j / p)
                        jp = np.mod(j + 1, p)
                        jm = np.mod(j - 1, p)
                        # right scattering
                        h[ind[j, 0, m, beta], ind[jp, 1, n, alpha]] += \
                            T2[alpha, beta] * fnm(z2[0], z2[1], n, m) * np.exp(
                                1j * ky * delta + 1j / 2 * ktheta * y0 + 1j * pi * q / p * (2 * j - 1))
                        # left scattering
                        h[ind[j, 0, m, beta], ind[jm, 1, n, alpha]] += \
                            T3[alpha, beta] * fnm(z3[0], z3[1], n, m) * np.exp(
                                -1j * ky * delta + 1j / 2 * ktheta * y0 + 1j * pi * q / p * (2 * j + 1))

    h = h + np.transpose(np.conj(h))
    argnan = np.argwhere(np.isnan(h))
    if argnan.size > 0:
        raise ValueError('Nan value caught in h')
    return ind, h


def plot_butterfly_from_points(pt_list, ylim=40, title=None, b_factor=None, min_alpha=0, max_alpha=6,
                               ax=None, markersize=1, mode='simple'):
    """
    Plots a Hofstadter diagram from a set of points. Automatically resizes to the maximal relevant energy level.

    :param pt_list: Array of FluxPoints.
    :param ylim: Limit for the y (energy) axis in meV.
    :param title: Title to use in the figure.
    :param b_factor: Ratio between flux and $B$ in Teslas.
    :param min_alpha: Minimal p/q.
    :param max_alpha: Maximal p/q.
    :param ax: Axis to plot the result on.
    :param markersize: Size of the points in the plot.

    :return: figure.
    """
    if ax is None:
        fig1, ax = plt.subplots()
        return_fig = True
    else:
        return_fig = False

    if title is not None:
        ax.set_title(
            title)  # f"kappa={params['kappa']}, theta = {np.rad2deg(params['theta'])}, max_l={max_l}, max_q={max_q}"
    if b_factor is not None:
        ax2 = ax.twiny()
    pqlist = np.array([pt.pq for pt in pt_list])

    b = pqlist * b_factor  # 2 * pi * pqlist / self.params['omega_m'] * hbar_e_angstrom2_to_tesla
    pq_points = np.concatenate([pt.pq_vec() for pt in pt_list])
    e_points = np.concatenate([pt.h_eigs.flatten() for pt in pt_list])
    max_y = max(e_points[np.abs(e_points) < ylim]) * 1.05
    min_y = min(e_points[np.abs(e_points) < ylim]) * 1.05
    ax.set_ylim(min_y, max_y)
    if mode == 'simple':
        ax.scatter(pq_points/6, e_points, marker='o', lw=0, s=markersize)
        if b_factor is not None:
            ax2.scatter(b, np.ones(len(b)), marker='')
        ax.set_xlabel(r'$\Phi$')
    else:
        raise ("Mode!= simple not yet supported.")

    ax2.set_xlabel('B [T]')
    ax.set_xlim(min_alpha/6, max_alpha/6)
    ax.minorticks_on()
    if return_fig:
        return fig1


@dataclasses.dataclass
class FluxPoint:
    """
    Class for information of the energy levels at a single p/q points.
    """
    p: int
    q: int
    pq: float
    mfield: float
    h_eigs: np.ndarray = np.array([])
    kx: np.ndarray = np.array([])
    ky: np.ndarray = np.array([])

    def __init__(self, p, q, mfield=None):
        self.p = p
        self.q = q
        self.pq = p / q
        self.mfield = mfield

    def below(self, e, k_point=None):
        """
        Returns the number of energy-levels below a certain level.
        :param e: Maximal absolute energy.
        :param k_point: If not none, measures the number of energy level at a specific k point.
        :return: Number of energy levels.
        """
        if k_point is None:
            return np.sum(np.abs(self.h_eigs) < e)
        else:
            return np.sum(np.abs(self.h_eigs[k_point]) < e)

    def append_k(self, kx, ky, h_eigs):
        """
        Appends the list of energy levels at a specific kx,ky point.
        :param kx: float kx.
        :param ky: float ky.
        :param h_eigs: List of energy levels.
        """
        np.append(self.kx, kx)
        np.append(self.ky, ky)
        if not self.h_eigs.size:
            self.h_eigs = h_eigs
        else:
            self.h_eigs = np.row_stack((self.h_eigs, h_eigs))

    def scatter_points(self):
        """Returns a list of (p/q,e) tuples, in 2*n np array."""
        return np.transpose(np.concatenate((np.ones(self.h_eigs.size) * self.pq, np.flatten(self.h_eigs))))

    def pq_vec(self):
        return np.ones(self.h_eigs.size) * self.pq

    def max_band_e(self, ylim):
        """Returns the maximal absolute energy in the p/q point.
        Assumes the energies of the band are smaller than ylim."""
        abs_h = np.abs(self.h_eigs)
        return max(abs_h[abs_h < ylim])


class LandauDiracModel:
    """
    Object for generating the Hofstadter butterfly.
    """

    def __init__(self, vel=.92e6, w=110, kappa=.8, d=1.42, theta=np.deg2rad(1.05), sublat_pot=0, layer_pot=0):
        """
        Initialize LandauDiracModel.

        :param vel: Dirac velocity in m/s.
        :param w: AB tunneling energy in meV.
        :param kappa: ratio between AA/AB tunneling energy.
        :param d: Distance between graphene atoms, in Angstrom.
        :param theta: Moire angle, in radians.
        :param sublat_pot: Optional: add potential between the A and B lattice cites.
        :param layer_pot: Optional: add potential between the Layers.
        """
        # lat_const = 1.42 a
        # dirac velocity = 1e6 m/s
        self.w1 = w * kappa
        self.w2 = w
        self.vel = vel
        self.w = w
        self._kappa = kappa
        self.d = d
        self.k_d = 4 * np.pi / 3 / np.sqrt(3) / d
        self._theta = theta
        self.sublat_pot = sublat_pot
        self.layer_pot = layer_pot

        self.ktheta = 2 * self.k_d * np.sin(theta / 2)
        self.omega_m = 12 * np.pi ** 2 / (np.sqrt(3) * self.ktheta ** 2)  # moire unit cell area

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self.theta = 2 * self.k_d * np.sin(theta / 2)
        self.theta = theta

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        self._kappa = kappa
        self.w1 = kappa * self.w

    def hamiltonian(self, kx, ky, p, q, max_l=10, estimate_max_l=True, return_index=False):
        """
        Hamiltonian of the Kronker product (p guiding)*(2 layer)*(n level)*(2 sublattice).
        Wraps the _hamiltonian_aux method.
        :param kx: Moire Brillouin zone kx.
        :param ky: Moire Brillouin zone kx.
        :param p,q: Flux numerator. Flux per unit cell=p/q/6.
        :param max_l: maximal energy level to include in the Hamiltonian.
        :param estimate_max_l: If true, estimates the maximal required energy level.
        :param return_index: If true, returns an array that converts the list [guiding,layer,level,sublattice] to
        row index of the hamiltonian.
        :return: Hamiltonian matrix.
        """
        max_l_d = max_l
        if estimate_max_l:
            max_l = int(max(max_l_d, self._get_max_l(p, q, max(self.w1, self.w2))))
        else:
            max_l = max_l_d
        ind, h = _hamiltonian_aux(kx, ky, p, q, max_l, self.theta,
                                  self.ktheta, self.vel, self.w1, self.w2, sublat_pot=self.sublat_pot,
                                  layer_pot=self.layer_pot)
        if return_index:
            return ind, h
        else:
            return h

    def kappa_perturbation_operator(self, kx, ky, p, q, max_l):
        """Generates the perturbation from kappa!=0 in a specific k point."""
        _, op = _hamiltonian_aux(kx, ky, p, q, max_l, self.theta,
                                 self.ktheta, 0, 1, 0, 0)
        return op

    def _get_max_l(self, p, q, w):
        """"""
        elambda = np.max((self.vel * self.ktheta * m_s_angstrom_to_mev, w))
        b = 2 * pi * p / q / self.omega_m
        mag_l = b ** (-1 / 2)
        wc = np.sqrt(2) * self.vel / mag_l * m_s_angstrom_to_mev
        return np.ceil(20 * (elambda / wc) ** 2)

    def get_butterfly(self, max_q=8, n_sub=3, mode='simple', ylim=500, estimate_max_l=True, check_convergence=True,
                      max_l=10, min_alpha=0., max_alpha=1., get_n_levels=False, levels_y_lim=None, save_images=True,
                      save_data=False,markersize=2):
        """
        plots the butterfly diagram. The flux per unit cell is p/q
        "simple" mode plots for p/q in (min_alpha,max_alpha)
        "inverted" mode plots from 1 to infinity, similar to the plots obtained in PhysRevB.84.035440.

        Optional: checks convergence using the get_g_eigvals check convergence procedure.


        :param max_q: Maximal denominator for flux fractions.
        :param n_sub: number of points in the mBZ for each flux value.
        :param mode: Simple by default. Other modes not currently supported.
        :param ylim: Maximal absolute value for th eenergy.
        :param estimate_max_l: If true, estimates the number of Landau levels required to ensure convergece,
        :param max_l: Used if estimate_max_l=False. Gives the number of Landau levels used in the Hamiltonians.
        :param min_alpha: Minimal value of p/q
        :param max_alpha: Maximal value of p/q
        :param get_n_levels: Bool. If true, plots also the number of energy levels whose absolute value is smaller than
            levels_y_lim
        :param levels_y_lim: Float array. List of energies to compate
        :param save_images: Bool. If true, saves the resulting figures as PDF.
        :param save_data: Bool. If true, saves the data as pickled array.
        :param markersize: size of the marker in the Hofstadter diagram.
        :return: If get_n_levels==False: Return Hofstadter figure.
            Else, returns fig1 as the Hofstadter plot and fig2 as the plot of number of energy levels.
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%m')  # for output figures

        # get range of flux values
        if mode == 'simple':
            plist, qlist, pqlist = gen_pqlist(max_q, min_alpha=min_alpha, max_alpha=max_alpha)
        elif mode == 'inverted':
            qlist, plist, _ = gen_pqlist(max_q)
            pqlist = plist / qlist
        pt_list = [FluxPoint(plist[i], qlist[i]) for i in range(len(plist))]

        for i in tqdm(range(len(qlist))):
            p = plist[i]
            q = qlist[i]
            pq = pqlist[i]
            k_theta = self.ktheta
            kx_rng = np.linspace(-1, 1, n_sub + 1)[:-1] * k_theta * np.sqrt(3 / 4) / 2
            ky_rng = np.linspace(-1, 1, n_sub + 1)[:-1] * k_theta / 4 / q

            for kx in kx_rng:
                for ky in ky_rng:
                    heig = self.get_h_eigvals(kx, ky, p, q, check_convergence=check_convergence, max_l=max_l,
                                              estimate_max_l=estimate_max_l)
                    pt_list[i].append_k(kx, ky, heig)

        fig1 = plot_butterfly_from_points(pt_list, ylim=ylim,
                                          b_factor=2 * pi / self.omega_m * hbar_e_angstrom2_to_tesla,
                                          min_alpha=min_alpha, max_alpha=max_alpha,markersize=markersize,
                                          title=f"kappa={self.kappa}, theta = {np.rad2deg(self.theta)}, max_l={max_l}, max_q={max_q}")
        if save_images:
            fname = path + f'figures/butterfly_{now}.pdf'
            if os.path.isfile(fname):
                for ii in range(100):
                    fname = path + f'figures/butterfly_{now}_{ii}.pdf'
                    if not os.path.isfile(fname):
                        break
            fig1.savefig(fname, dpi=200)

        if save_data:
            fname = path + f'out_data_{now}'
            if os.path.isfile(fname):
                for ii in range(100):
                    fname = path + f'out_data_{now}_{ii}'
                    if not os.path.isfile(fname):
                        break
            np.save(fname, np.array(pt_list, dtype=object))

        if get_n_levels:
            fig2 = plt.figure()
            for i, level_lim in enumerate(levels_y_lim):
                n_levels = [pt.below(level_lim) / pt.q / 6 for pt in pt_list]
                plt.scatter(pqlist, n_levels, label=f'E0={levels_y_lim[i]}', marker='x')
            plt.xlabel('p/q')
            plt.ylabel('Number of states per unit cell with |E|<E0')
            plt.minorticks_on()
            plt.grid(b=True, which='both', color='#999999', linestyle='-')
            plt.legend()
            if save_images:
                fig2.savefig(path + f'figures/levels_below_{now}.pdf')

        if get_n_levels:
            return fig1, fig2
        else:
            return fig1

    def _check_convergence(self, p, q, min_l, max_l, dl, ylim=100):
        """plots the energy eigenvalues as function of number of levels.
        Debugging function, do not use."""
        energies = np.array([])
        l_points = np.array([])

        for l in tqdm(range(min_l, max_l, dl)):
            h = self.hamiltonian(0, 0, p, q, max_l=l, estimate_max_l=False)
            eigsi = linalg.eigvalsh(h)
            energies = np.concatenate((energies, eigsi))
            l_points = np.concatenate((l_points, np.ones(len(eigsi)) * l))

        plt.figure()
        plt.scatter(l_points, energies, marker=',', lw=0, s=1)
        plt.ylim(-ylim, ylim)

    def _plot_minimal_eigenvectors(self, p, q, max_l, n_eigs=4):
        """debugging function: plots the absolute value square of the lowest energy levels eigenfunctions"""
        h = self.hamiltonian(0, 0, p, q, max_l, estimate_max_l=False)
        eigvals, eigvecs = linalg.eigh(h)
        sorted_eigvals = np.sort(np.abs(eigvals))
        eigvecs = np.transpose(eigvecs)
        plt.figure()
        for i in np.argwhere(np.abs(eigvals) <= sorted_eigvals[n_eigs + 1]):
            plt.plot(np.abs(eigvecs[i][0]) ** 2, label=str(eigvals[i][0]))
        plt.legend()

    def get_dos(self, p, q, ylim=100, n_sub=10, estimate_max_l=False, max_l=100, n_bins=100):
        """
        Gets the DOS(energy) for a given p/q fraction.
        :param p,q: Flux numerator/denominator.
        :param ylim: Maximal energy to consider.
        :param n_sub: Number of points in the BZ grid. The grid is n_sub*n_sub.
        :param estimate_max_l: If true, estimates the number of Landau levels for the convergence of the.
        :param max_l: Number of energy levels if estimate_max_l=False.
        :param n_bins: Number of bins for output.
        :return: energies,dos: Numpy arrays of list of energies and associated DOS.
        """
        k_theta = self.ktheta
        kx_rng = np.linspace(-1, 1, n_sub) * k_theta * np.sqrt(3 / 4) / 2
        ky_rng = np.linspace(-1, 1, n_sub) * k_theta / 4 / q
        eigs_list = np.array([])
        self.get_h_eigvals(0, 0, p, q, ylim=ylim, estimate_max_l=estimate_max_l, max_l=max_l)
        for kx in tqdm(kx_rng):
            for ky in ky_rng:
                h = self.hamiltonian(kx, ky, p, q, max_l=max_l, estimate_max_l=estimate_max_l)
                heig = linalg.eigvalsh(h)
                eigs_list = np.concatenate((eigs_list, heig[np.abs(heig) < ylim]))

        eigs_list = np.sort(eigs_list)
        min_e = np.min(eigs_list)
        max_e = np.max(eigs_list)
        energies = np.linspace(min_e - (max_e - min_e) * .1, max_e + (max_e - min_e) * .1, n_bins)
        kernel_size = (max_e - min_e) / (24 * n_sub ** 2)
        dos = np.array(
            [np.sum([np.exp(-((e - eig) ** 2 / kernel_size)) for eig in eigs_list]) / (len(kx_rng) * len(ky_rng) * q)
             for e in energies])
        return energies, dos

    def plot_dos(self, p, q, ylim=100, n_sub=10, estimate_max_l=False, max_l=100, n_bins=100):
        """
        Plots the DOS(energy) for a given p/q fraction.
        :param p,q: Flux numerator/denominator.
        :param ylim: Maximal energy to consider.
        :param n_sub: Number of points in the BZ grid. The grid is n_sub*n_sub.
        :param estimate_max_l: If true, estimates the number of Landau levels for the convergence of the.
        :param max_l: Number of energy levels if estimate_max_l=False.
        :param n_bins: Number of bins for output.
        :return: energies,dos: Numpy arrays of list of energies and associated DOS.
        """
        energies, dos = self.get_dos(p, q, ylim=ylim, n_sub=n_sub, estimate_max_l=estimate_max_l, max_l=max_l,
                                     n_bins=n_bins)
        plt.figure()
        plt.plot(energies, dos)

    def get_h_eigvals(self, kx, ky, p: int, q: int, check_convergence=True, ylim=20, **kwargs):
        """gets the eigenvalues of h, optionally uses a procedure to check the convergence of the energy.

        convergence test: checks that energy in the localized in the first relevant_levels levels."""
        h = self.hamiltonian(kx, ky, p, q, **kwargs)
        max_l = int(len(h) / 2 / 2 / p)
        relevant_levels = 3 / 4
        convergence_factor = .95
        # checks that the lowest 50% of landau levels hold 90% of the vector.
        if check_convergence:
            vec_range = np.concatenate(
                [2 * max_l * i + np.arange(int(relevant_levels * 2 * max_l)) for i in range(2 * int(p))])
            eigvals, eigvecs = linalg.eigh(h)
            eigvecs = np.transpose(eigvecs)
            for i in range(len(eigvals)):
                if np.abs(eigvals[i]) < ylim:
                    if np.sum(np.abs(eigvecs[i][vec_range]) ** 2) < convergence_factor:
                        warnings.warn(
                            f"Convergence error: cannot ensure convergence for energy {eigvals[i]} at p,q={(p, q)}")
        else:
            eigvals = linalg.eigvalsh(h)
        return eigvals

    def plot_bandstructure(self, p: int, q: int, ylim=100, n_sub=4, max_l=10, estimate_max_l=False):
        """ Plots the band structure in a region of k.
        Not very useful.

        Parameters
        ----------
        p,q:
            flux fraction, with the flux per unit cell Phi=p/q/6.
        ylim:
            region maximal energy of the bands to be plotted.
        n_sub:
            number of k points per path segment in k space

        """

        fig, ax = plt.subplots()
        ktheta = self.ktheta
        k_list = np.concatenate((np.transpose([np.zeros(n_sub), np.linspace(0, ktheta / 2 / q, n_sub)])[:-1],
                                 np.transpose([np.linspace(0, 1, n_sub) * ktheta * np.sqrt(3 / 4),
                                               ktheta / 2 / q * np.ones(n_sub)])[:-1],
                                 np.transpose([np.linspace(1, -1, n_sub) * ktheta * np.sqrt(3 / 4),
                                               np.linspace(1, -1, n_sub) * ktheta / 2 / q])[:-1],
                                 np.transpose([- np.ones(n_sub) * ktheta * np.sqrt(3 / 4),
                                               np.linspace(-1, 0, n_sub) * ktheta / 2 / q])[:-1],
                                 np.transpose([np.linspace(-1, 0, n_sub) * ktheta * np.sqrt(3 / 4), np.zeros(n_sub)])
                                 )
                                )

        if estimate_max_l:
            max_l = int(max(max_l, self._get_max_l(p, q, max(self.w1, self.w2))))

        eigs_list = np.zeros((len(k_list), 2 * 2 * p * (max_l + 1)))
        for i, k in enumerate(k_list):
            h = self.hamiltonian(k[0], k[1], p, q, max_l=max_l, estimate_max_l=estimate_max_l)
            k_eigs = linalg.eigvalsh(h)
            eigs_list[i, :] = np.sort(np.real(k_eigs))

        nlist = np.arange(len(eigs_list))

        for i in range(eigs_list.shape[1]):
            plt.plot(nlist, eigs_list[:, i])

        plt.ylim(-ylim, ylim)
        ax.set_xticks(np.concatenate((np.arange(1), np.arange(1, 5) * (n_sub - 1), [len(k_list) - 1])))
        ax.set_xticklabels(['(0,0)',
                            r'$(0,\frac{1}{2q})k_\theta$',
                            r'$(\frac{\sqrt{3}}{2},\frac{1}{2q})k_\theta$',
                            r'$(-\frac{\sqrt{3}}{2},-\frac{1}{2q})k_\theta$',
                            r'$(-\frac{\sqrt{3}}{2},0)k_\theta$',
                            '(0,0)'])
        ax.set_title(f'p={p},q={q}')
        ax.set_ylabel('E [meV]')

    def plot_bandwidth(self, p, q, theta_min, theta_max, n_theta, max_l=200, ylim=20, use_degrees=True):
        """Plots the bandwidth at a certain magnetic field as a function of angle theta"""
        theta_rng = np.linspace(theta_min, theta_max, n_theta)
        if use_degrees:
            theta_rng = np.deg2rad(theta_rng)
        bw_rng = np.zeros(len(theta_rng))
        for i, theta in tqdm(enumerate(theta_rng)):
            self.set_theta(theta)
            eigs = self.get_h_eigvals(0, 0, p, q, max_l=max_l)
            band = eigs[np.abs(eigs) < ylim]
            bw_rng[i] = max(band) - min(band)
        plt.figure()
        plt.plot(np.rad2deg(theta_rng), bw_rng)
        plt.xlabel('theta(deg)')
        plt.ylabel('bw [meV]')


if __name__ == '__main__':
    min_alpha = .1
    max_alpha = 8
    theta = np.deg2rad(1.20)  # 1.0434
    kappa = .0
    model = LandauDiracModel(theta=theta, kappa=kappa, sublat_pot=1)
    model.get_butterfly(max_q=2, n_sub=1, ylim=30, estimate_max_l=True,
                        min_alpha=min_alpha, max_alpha=max_alpha,
                        save_images=False, save_data=False, markersize=3,
                        get_n_levels=True, levels_y_lim=[0.0001, 5])
    plt.show()
