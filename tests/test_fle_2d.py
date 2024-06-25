import matplotlib.pyplot as plt
import time
from os.path import exists
from scipy.io import savemat
from fle_2d import FLEBasis2D
import numpy as np
from scipy.io import loadmat



def main():

    # test 1: Verify that code agrees with dense matrix mulitplication
    print("test 1")
    test1_fle_vs_dense()
    test1_complex_fle_vs_dense()

    # test 2: verify that code can lowpass
    print("test 2")
    test2_fle_lowpass()

    # test 3: part timing 
    print("test 3")
    test3_part_timing()

    # test 4: Check that we can rotate an image
    print("test 4")
    test4_check_rotate()

    # test5: Check that we can convolve with radial functions efficiently
    print("test 5")
    test5_check_radialconv()

    # test6: check that it works for tensor inputs
    print("test 6")
    test6_check_tensor_input_works()
    test6_complex_check_tensor_input_works()
    # test7: expand error
    print("test 7")
    test7_expand_error_test()

    # test8: verify that the code works for images with odd dimensions
    print("test 8")
    test8_fle_vs_dense_odd()
    test8_complex_fle_vs_dense_odd()

    plt.show()
    return

def test1_fle_vs_dense():

    Ls = 32 + 32*np.arange(2)
    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in Ls:
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    erra = np.zeros(n)
    errx = np.zeros(n)
    for i in range(n):
        erra[i], errx[i] = test1_fle_vs_dense_helper(ls[i], epss[i])

    # make {tab:accuracy}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $\\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            ls[i],
            "&",
            "{:12.5e}".format(epss[i]),
            "&",
            "{:12.5e}".format(erra[i]),
            "&",
            "{:12.5e}".format(errx[i]),
            "\\\\",
        )
        if i % len(Ls) == len(Ls) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test1_fle_vs_dense_helper(L, eps):

    # Parameters
    # Bandlimit scaled so that L is maximum suggested bandlimit

    # Basis pre-computation
    bandlimit = L
    fle = FLEBasis2D(L, bandlimit, eps)
    t1 = time.time()

    # Create 
    B = fle.create_denseB(numthread=10)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))
    # x = B[:,2]
    # evaluate_t
    a_dense = B.T @ x
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # printt

    erra = relerr(a_dense, a_fle)
    errx = relerr(xlow_dense, xlow_fle)

    # plt.plot(np.real(a_dense))
    # plt.plot(np.real(a_fle))

    # plt.figure()
    # plt.plot(np.real(a_dense))
    # plt.plot(np.real(a_fle))

    # plt.figure()
    # plt.plot(np.real(a_fle-a_dense))

    # plt.plot(np.abs(a_dense))
    # plt.plot(np.abs(a_fle))
    # plt.show()

    return erra, errx

def test1_complex_fle_vs_dense():

    Ls = 32 + 32*np.arange(2)
    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in Ls:
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    erra = np.zeros(n)
    errx = np.zeros(n)
    for i in range(n):
        erra[i], errx[i] = test1_complex_fle_vs_dense_helper(ls[i], epss[i])

    # make {tab:accuracy}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $\\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            ls[i],
            "&",
            "{:12.5e}".format(epss[i]),
            "&",
            "{:12.5e}".format(erra[i]),
            "&",
            "{:12.5e}".format(errx[i]),
            "\\\\",
        )
        if i % len(Ls) == len(Ls) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test1_complex_fle_vs_dense_helper(L, eps):

    # Parameters
    # Bandlimit scaled so that L is maximum suggested bandlimit

    # Basis pre-computation
    bandlimit = L
    fle = FLEBasis2D(L, bandlimit, eps, mode="complex")
    t1 = time.time()

    # Create 
    B = fle.create_denseB(numthread=10)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    # evaluate_t
    a_dense = np.conj(B.T) @ x
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # printt

    erra = relerr(a_dense, a_fle)
    errx = relerr(xlow_dense, xlow_fle)

    return erra, errx

def test2_fle_lowpass():

    # Parameters
    # Use L x L images
    L = 128
    # Bandlimit scaled so that L is maximum suggested bandlimit
    bandlimit = L
    # Relative error compared to dense matrix method
    eps = 1e-14

    # Basis pre-computation
    fle = FLEBasis2D(L, bandlimit, eps)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    imshow_disk(fle, "L=" + str(L) + " original", x)

    # basic low pass
    for k in range(4):
        bandlimit = L // (2**k)
        a_fle = fle.evaluate_t(x)
        a_low = fle.lowpass(a_fle, bandlimit)
        xlow = fle.evaluate(a_low)
        print("bandlimit", bandlimit)
        print("num nonzero coeff", np.sum(a_low != 0))
        imshow_disk(
            fle, "FLEBasis2D L=" + str(L) + " bandlimit=" + str(bandlimit), xlow
        )

def test3_part_timing():

    nr = 1  # number of trials
    Ls = (512,)
    eps = 1e-7
    n = len(Ls)

    dts = np.zeros((n, 6, nr))
    for i in range(n):
        L = Ls[i]
        bandlimit = L
        fle = FLEBasis2D(L, bandlimit, eps)
        # load example image
        datafile = "test_images/data_L=" + str(L) + ".mat"
        data = loadmat(datafile)
        x = data["x"]
        x = x / np.max(np.abs(x.flatten()))
        x = x.reshape((L**2, 1))
        for j in range(nr):
            dts[i, :, j] = test3_helper(L, fle, x)

    dts_avg = np.sum(dts, axis=2) / nr

    # make {tab:accuracy}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $dt_1$ & $dt_2$ & dt_3 & dt_1^H & dt_2^h & dt_3^h \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Ls[i],
            "&",
            "{:10.3e}".format(dts_avg[i, 0]),
            "&",
            "{:10.3e}".format(dts_avg[i, 1]),
            "&",
            "{:10.3e}".format(dts_avg[i, 2]),
            "&",
            "{:10.3e}".format(dts_avg[i, 3]),
            "&",
            "{:10.3e}".format(dts_avg[i, 4]),
            "&",
            "{:10.3e}".format(dts_avg[i, 5]),
            "\\\\",
        )
        if i % len(Ls) == len(Ls) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test3_helper(L, fle, x):

    # Get ready for Step 1
    L = fle.L
    f = np.copy(x).reshape(fle.L, fle.L)
    f[fle.idx] = 0
    f = f.flatten()

    # Step 1. {sec:fast_details}
    t0 = time.time()
    z = fle.step1(f)
    t1 = time.time()
    dt1 = t1 - t0

    # Step 2. {sec:fast_details}
    t0 = time.time()
    b = fle.step2(z)
    t1 = time.time()
    dt2 = t1 - t0

    # Step 3: {sec:fast_details}
    t0 = time.time()
    a = fle.step3(b)
    t1 = time.time()
    dt3 = t1 - t0

    # Step 3 adjoint
    t0 = time.time()
    b = fle.step3_H(a)
    t1 = time.time()
    dt1H = t1 - t0

    # Step 2 adjoint
    t0 = time.time()
    z = fle.step2_H(b)
    t1 = time.time()
    dt2H = t1 - t0

    # Step 1 adjoint
    t0 = time.time()
    f = fle.step1_H(z)
    t1 = time.time()
    dt3H = t1 - t0

    dts = [dt1, dt2, dt3, dt1H, dt2H, dt3H]
    return dts





def test4_check_rotate():

    # Use L x L images
    L = 128
    bandlimit = L
    eps = 1e-7

    # Basis pre-computation
    fle = FLEBasis2D(L, bandlimit, eps, mode="complex")

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    a = fle.evaluate_t(x)
    xlow = fle.evaluate(a)

    theta = np.pi / 3
    a_rot = fle.rotate(a, theta)
    xlow_rot = fle.evaluate(a_rot)

    imshow_disk(fle, "FLEBasis2D xlow", xlow)
    imshow_disk(fle, "FLEBasis2D xlow rot by pi/3", xlow_rot)

    return


def test5_check_radialconv():

    # Use L x L images
    L = 128
    bandlimit = L
    eps = 1e-7

    # Basis pre-computation
    fle = FLEBasis2D(L, bandlimit, eps, mode="complex")

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    a = fle.evaluate_t(x)
    xlow = fle.evaluate(a)

    # Load example CTF
    t0 = time.time()
    datafile = "test_images/ctf_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    ctf = data["a"]
    ctf = ctf / np.max(np.abs(ctf.flatten()))
    ctf = ctf.reshape((L**2, 1))
    t1 = time.time()
    dt = t1 - t0

    a = fle.evaluate_t(x)
    xlow = fle.evaluate(a)

    a_ctf = fle.evaluate_t(ctf)
    ctflow = fle.evaluate(a_ctf)

    imshow_disk(fle, "ctf", ctflow)
    imshow_disk(fle, "x", xlow)

    # Convolve using FFT
    ctflow_pad = np.zeros((2 * L, 2 * L))
    xlow_pad = np.zeros((2 * L, 2 * L))
    ctflow_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = ctflow
    xlow_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L] = xlow

    ctf_shift = np.copy(ctflow_pad).reshape(2 * L, 2 * L)
    xlow_shift = np.copy(xlow_pad).reshape(2 * L, 2 * L)
    ctf_shift = np.fft.fftshift(ctf_shift)
    xlow_shift = np.fft.fftshift(xlow_shift)
    xlow_conv_fft_pad = np.fft.fftshift(
        np.fft.ifft2(np.fft.fft2(ctf_shift) * np.fft.fft2(xlow_shift))
    )
    xlow_conv_fft = xlow_conv_fft_pad[L // 2 : L // 2 + L, L // 2 : L // 2 + L]
    imshow_disk(fle, "FFT xlow_conv", xlow_conv_fft)

    # Convolve using coefficients
    a_conv = fle.radialconv(a, ctflow)
    xlow_conv = fle.evaluate(a_conv)

    imshow_disk(fle, "FLEBasis2D xlow_conv", xlow_conv)
    imshow_disk(
        fle, "FLEBasis2D xlow_conv - FFT xlow_conv", xlow_conv - xlow_conv_fft
    )

    err = relerr(xlow_conv_fft, xlow_conv)
    print("Relative error FFT xlow_conv vs. FLEBasis2D xlow_conv", err)

    mask = fle.rs < 0.7
    err = relerr(mask * xlow_conv_fft, mask * xlow_conv)
    print(
        "Relative error FFT (rs<.7)*xlow_conv vs. FLEBasis2D (rs<.7)*xlow_conv",
        err,
    )

    return


def test6_check_tensor_input_works():

    # Use L x L images
    L = 128
    bandlimit = L
    eps = 1e-7
    N = 3

    # Basis pre-computation
    fle = FLEBasis2D(L, bandlimit, eps)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    a = fle.evaluate_t(x)
    xlow = fle.evaluate(a)

    # Create a three dimensional array with rotated images
    X = np.zeros((N, L, L), dtype=np.float64, order="C")
    X_check = np.zeros((N, L, L), dtype=np.float64, order="C")
    a_check = np.zeros((N, fle.ne))
    for i in range(N):
        theta = 2 * np.pi * i / N
        b_rot = fle.rotate_multipliers(theta)
        a_rot = fle.c2r @ (b_rot * (fle.r2c @ a))
        X[i, :, :] = fle.evaluate(a_rot)
        a_check[i, :] = fle.evaluate_t(X[i, :, :])
        X_check[i, :, :] = fle.evaluate(a_check[i, :])

    # Test the tensor input
    a = fle.evaluate_t(X)
    X = fle.evaluate(a)

    # Print error
    err = relerr(a, a_check)
    print("FLEBasis2D coefficient tensor error", err)
    err = relerr(X, X_check)
    print("FLEBasis2D image tensor error", err)

    return

def test6_complex_check_tensor_input_works():

    # Use L x L images
    L = 128
    bandlimit = L
    eps = 1e-7
    N = 3

    # Basis pre-computation
    fle = FLEBasis2D(L, bandlimit, eps, mode="complex")

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    a = fle.evaluate_t(x)
    xlow = fle.evaluate(a)

    # Create a three dimensional array with rotated images
    X = np.zeros((N, L, L), dtype=np.float64, order="C")
    X_check = np.zeros((N, L, L), dtype=np.float64, order="C")
    a_check = np.zeros((N, fle.ne),dtype=np.complex128)
    for i in range(N):
        theta = 2 * np.pi * i / N
        b_rot = fle.rotate_multipliers(theta)
        a_rot = fle.c2r @ (b_rot * (fle.r2c @ a))
        X[i, :, :] = fle.evaluate(a_rot)
        a_check[i, :] = fle.evaluate_t(X[i, :, :])
        X_check[i, :, :] = fle.evaluate(a_check[i, :])

    # Test the tensor input
    a = fle.evaluate_t(X)
    X = fle.evaluate(a)

    # Print error
    err = relerr(a, a_check)
    print("FLEBasis2D complex coefficient tensor error", err)
    err = relerr(X, X_check)
    print("FLEBasis2D complex image tensor error", err)

    return


def test7_expand_error_test():

    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in (32, 64):
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    err = np.zeros(n)
    for i in range(n):
        err[i] = test7_helper(ls[i], epss[i])

    # make {tab:accuracy}
    print("expand test")
    for i in range(n):
        print(
            ls[i],
            " ",
            "{:12.5e}".format(epss[i]),
            " ",
            "{:12.5e}".format(err[i]),
        )


def test7_helper(L, eps):

    # Parameters
    # Bandlimit scaled so that L is maximum suggested bandlimit

    # Basis pre-computation
    bandlimit = L
    fle = FLEBasis2D(L, bandlimit, eps)
    t1 = time.time()

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))
    x = fle.evaluate(fle.evaluate_t(x))

    # evaluate_t
    a0 = fle.expand(x)
    x0 = fle.evaluate(a0)
    err = relerr(x, x0)

    return err



def imshow_disk(fle, figtitle, x):
    L = fle.L
    vis = np.real(np.copy(x))
    vis = vis.reshape((L, L))
    vis[fle.idx] = np.nan
    plt.figure()
    plt.imshow(vis, interpolation="none")
    plt.title(figtitle)
    plt.colorbar()


def relerr(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.linalg.norm(x - y) / np.linalg.norm(x)


def test8_fle_vs_dense_odd():

    Ls = (33,65)
    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in Ls:
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    erra = np.zeros(n)
    errx = np.zeros(n)
    for i in range(n):
        erra[i], errx[i] = test8_fle_vs_dense_odd_helper(ls[i], epss[i])

    # make {tab:accuracy}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $\\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            ls[i],
            "&",
            "{:12.5e}".format(epss[i]),
            "&",
            "{:12.5e}".format(erra[i]),
            "&",
            "{:12.5e}".format(errx[i]),
            "\\\\",
        )
        if i % len(Ls) == len(Ls) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test8_fle_vs_dense_odd_helper(L, eps):

    # Parameters
    # Bandlimit scaled so that L is maximum suggested bandlimit

    # Basis pre-computation
    bandlimit = L
    fle = FLEBasis2D(L, bandlimit, eps)
    t1 = time.time()

    # Create 
    B = fle.create_denseB(numthread=10)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    # evaluate_t
    a_dense = B.T @ x
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # printt

    erra = relerr(a_dense, a_fle)
    errx = relerr(xlow_dense, xlow_fle)

    return erra, errx

def test8_complex_fle_vs_dense_odd():

    Ls = (33,65)
    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in Ls:
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    erra = np.zeros(n)
    errx = np.zeros(n)
    for i in range(n):
        erra[i], errx[i] = test8_complex_fle_vs_dense_odd_helper(ls[i], epss[i])

    # make {tab:accuracy}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $\\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            ls[i],
            "&",
            "{:12.5e}".format(epss[i]),
            "&",
            "{:12.5e}".format(erra[i]),
            "&",
            "{:12.5e}".format(errx[i]),
            "\\\\",
        )
        if i % len(Ls) == len(Ls) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test8_complex_fle_vs_dense_odd_helper(L, eps):

    # Parameters
    # Bandlimit scaled so that L is maximum suggested bandlimit

    # Basis pre-computation
    bandlimit = L
    fle = FLEBasis2D(L, bandlimit, eps, mode="complex")
    t1 = time.time()

    # Create 
    B = fle.create_denseB(numthread=10)

    # load example image
    datafile = "test_images/data_L=" + str(L) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = x.reshape((L**2, 1))

    # evaluate_t
    a_dense = np.conj(B.T) @ x
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # printt

    erra = relerr(a_dense, a_fle)
    errx = relerr(xlow_dense, xlow_fle)

    return erra, errx



if __name__ == "__main__":
    main()
