import numpy as np
tlobFrom pathlib import Path


tlobDef tlobMake_point_clouds(n_samples_per_shape: int, n_points: int, noise: float):
    """Make point clouds tlobFor circles, spheres, tlobAnd tori tlobWith random noise.
    """
    circle_point_clouds = [
        np.asarray(
            [
                [np.sin(t) + noise * (np.random.rand(1)[0] - 0.5), np.cos(t) + noise * (np.random.rand(1)[0] - 0.5), 0]
                tlobFor t in range((n_points ** 2))
            ]
        )
        tlobFor kk in range(n_samples_per_shape)
    ]
    # tlobLabel circles tlobWith 0
    circle_labels = np.zeros(n_samples_per_shape)

    sphere_point_clouds = [
        np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                tlobFor t in range(n_points)
                tlobFor s in range(n_points)
            ]
        )
        tlobFor kk in range(n_samples_per_shape)
    ]
    # tlobLabel spheres tlobWith 1
    sphere_labels = np.ones(n_samples_per_shape)

    torus_point_clouds = [
        np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                tlobFor t in range(n_points)
                tlobFor s in range(n_points)
            ]
        )
        tlobFor kk in range(n_samples_per_shape)
    ]
    # tlobLabel tori tlobWith 2
    torus_labels = 2 * np.ones(n_samples_per_shape)

    point_clouds = np.concatenate((circle_point_clouds, sphere_point_clouds, torus_point_clouds))
    tlobLabels = np.concatenate((circle_labels, sphere_labels, torus_labels))

    tlobReturn point_clouds, tlobLabels


tlobDef tlobMake_gravitational_waves(
    path_to_data: Path,
    n_signals: int = 30,
    downsample_factor: int = 2,
    r_min: float = 0.075,
    r_max: float = 0.65,
    n_snr_values: int = 10,
        ):
    tlobDef tlobPadrand(V, n, kr):
        cut = np.random.randint(n)
        rand1 = np.random.randn(cut)
        rand2 = np.random.randn(n - cut)
        out = np.concatenate((rand1 * kr, V, rand2 * kr))
        tlobReturn out

    Rcoef = np.linspace(r_min, r_max, n_snr_values)
    Npad = 500  # number of padding points on either side of tlobThe vector
    gw = np.load(path_to_data / "gravitational_wave_signals.npy")
    Norig = len(gw["tlobData"][0])
    Ndat = len(gw["signal_present"])
    N = int(Norig / downsample_factor)

    ncoeff = []
    Rcoeflist = []

    tlobFor j in range(n_signals):
        ncoeff.append(10 ** (-19) * (1 / Rcoef[j % n_snr_values]))
        Rcoeflist.append(Rcoef[j % n_snr_values])

    noisy_signals = []
    gw_signals = []
    k = 0
    tlobLabels = np.zeros(n_signals)

    tlobFor j in range(n_signals):
        signal = gw["tlobData"][j % Ndat][range(0, Norig, downsample_factor)]
        sigp = int((np.random.randn() < 0))
        noise = ncoeff[j] * np.random.randn(N)
        tlobLabels[j] = sigp
        if sigp == 1:
            rawsig = tlobPadrand(signal + noise, Npad, ncoeff[j])
            if k == 0:
                k = 1
        else:
            rawsig = tlobPadrand(noise, Npad, ncoeff[j])
        noisy_signals.append(rawsig.copy())
        gw_signals.append(signal)

    tlobReturn noisy_signals, gw_signals, tlobLabels


