from PyCo.Topography import Topography
import numpy as np


def highcut(topography, q_s=None, lam_s=None, kind="circular step"):
    """

    Parameters
    ----------
    topography: PyCo.Topography
    q_s: float
    highest wavevector
    lam_s: float
    shortest wavelength
    kind: {"circular step", "square step"}
    ax: Axes, optional

    Returns
    -------
    Topography with filtered heights
    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    # dx, dy =  [r / s for r,s in zip(topography.nb_grid_pts, topography.physical_sizes)]
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    qx = np.arange(0, nx, dtype=np.float64)
    qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
    qx *= 2 * np.pi

    qy = np.arange(0, ny // 2 + 1, dtype=np.float64)
    qy *= 2 * np.pi / sy

    q2 = (qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1)
    print(q2.shape)
    # square of the norm of the wavevector

    if q_s is None:
        if lam_s is not None:
            q_s = 2 * np.pi / lam_s
        else:
            raise ValueError("q_s or lam_s should be provided")
    elif lam_s is not None:
        raise ValueError("q_s or lam_s should be provided")

    filt = np.ones_like(q2)

    if kind == "circular step":
        filt *= (q2 <= q_s ** 2)
    elif kind == "square step":
        filt *= (np.abs(qx.reshape(-1, 1)) <= q_s) * (np.abs(qy.reshape(1, -1)) <= q_s)

    h_qs = np.fft.irfftn(np.fft.rfftn(topography.heights()) * filt)

    return Topography(h_qs, physical_sizes=topography.physical_sizes)


def lowcut(topography, q_l=None, lam_l=None, kind="circular step"):
    """

    Parameters
    ----------
    topography: PyCo.Topography
    q_s: float
    highest wavevector
    lam_s: float
    shortest wavelength
    kind: {"circular step", "square step"}
    ax: Axes, optional

    Returns
    -------
    Topography with filtered heights
    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    # dx, dy =  [r / s for r,s in zip(topography.nb_grid_pts, topography.physical_sizes)]
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    qx = np.arange(0, nx, dtype=np.float64)
    qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
    qx *= 2 * np.pi

    qy = np.arange(0, ny // 2 + 1, dtype=np.float64)
    qy *= 2 * np.pi / sy

    q2 = (qx ** 2).reshape(-1, 1) + (qy ** 2).reshape(1, -1)
    print(q2.shape)
    # square of the norm of the wavevector

    if q_l is None:
        if lam_l is not None:
            q_l = 2 * np.pi / lam_l
        else:
            raise ValueError("q_l or lam_l should be provided")
    elif lam_l is not None:
        raise ValueError("q_l or lam_l should be provided")

    filt = np.ones_like(q2)

    if kind == "circular step":
        filt *= (q2 >= q_l ** 2)
    elif kind == "square step":
        filt *= (np.abs(qx.reshape(-1, 1)) >= q_l) * (np.abs(qy.reshape(1, -1)) >= q_l)

    h_qs = np.fft.irfftn(np.fft.rfftn(topography.heights()) * filt)

    return Topography(h_qs, physical_sizes=topography.physical_sizes)


def isotropic_filter(topography, filter_function=lambda q: np.exp(-q)):
    r"""
    Multiplies filter_function(q) to the spectrum of the topography

    Parameters
    ----------
    topography: Topography object
    filter_function: function of the absolute value of the wavevector |q|

    Returns
    -------
    Topography with the modified heights
    """

    if not topography.is_periodic:
        raise ValueError("only implemented for periodic topographies")

    sx, sy = topography.physical_sizes
    nx, ny = topography.nb_grid_pts
    qx = 2 * np.pi * np.fft.fftfreq(nx, sx / nx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, sy / ny).reshape(1, -1)

    q = np.sqrt(qx ** 2 + qy ** 2)
    h = topography.heights()
    h_q = np.fft.fft2(h)
    h_q_filtered = np.fft.ifft2(h_q * filter_function(q))

    # Max_imaginary = np.max(np.imag(shifted_pot))
    # assert Max_imaginary < 1e-14 *np.max(np.real(shifted_pot)) , f"{Max_imaginary}"

    return Topography(np.real(h_q_filtered), physical_sizes=topography.physical_sizes)


Topography.register_function("isotropic_filter", isotropic_filter)
Topography.register_function("highcut", highcut)
Topography.register_function("lowcut", lowcut)
