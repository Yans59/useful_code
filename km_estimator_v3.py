# Using KM estimator to calculate mean and err (for data with upperlimits)
# reference: Feigelson & Nelson 1985, ApJ 293, 192 (Eq 5-8, Eq 13-15)
#---
# CX  1997.08.15 v0  (Xu et al. 1998, ApJ 508, 576)
# CKX 2018.10.25 v1
# CKX 2020.08.17 v2
#---
# HC  2020.08.20 v.matlab.0
#   translate to matlab
#   some minor change
#---
# YSL 2025.01.12 v.python.3
#   translate to python
# YSL 2025.08.29 v.python.3
#   Add percentile
# YSL 2025.09.03 v.python.3
#   Add weighted non-detections
#---
# the type of data and flag should be numpy.array
# flag: 1 = detect, 0 = upperlimit

import numpy as np

def km_estimator(data, flag, alpha_weight=True):
    """
    alpha_weight : bool
        True  → cumulative survial probability for upper limits
        False → standard KM
    """
    data = np.asarray(data, dtype=float)
    flag = np.asarray(flag, dtype=int)
    ndata = len(data)

    data_max = np.max(data)
    data_rs = data_max - data  # left censored to right censored
    order = np.argsort(data_rs)
    data_sort = data_rs[order]
    flag_sort = flag[order]

    # set detection of the first and last ones
    flag_sort[[0, ndata - 1]] = [1, 1]

    # survival function
    ss = np.zeros(ndata, dtype=float)
    ss[0] = 1.0
    for i in range(1, ndata):
        r = ndata - i + 1
        if alpha_weight:
            ss[i] = ss[i - 1] * (1 - 1.0 / r) ** flag_sort[i - 1] * (1 - ss[i - 1] / r) ** (1 - flag_sort[i - 1])
        else:
            ss[i] = ss[i - 1] * (1 - 1.0 / r) ** flag_sort[i - 1] 

    # mean value
    ay = np.sum(ss[1:] * np.diff(data_sort))
    mean_val = data_max - ay

    # std
    ex2 = 0.0
    for j in range(1, ndata):
        if flag_sort[j - 1] == 1:
            a1 = np.sum(ss[j:] * np.diff(data_sort[j - 1:]))
            ex2 += a1 ** 2 / (ndata - j + 1) / (ndata - j)
    se_val = np.sqrt(ex2)

    return mean_val, se_val


def km_estimator_per(data, flag, percentiles=[0.5], alpha_weight=True):
    """
    percentiles: a list of percentile
    
    alpha_weight : bool
        True  → cumulative survial probability for upper limits
        False → standard KM
    """
    data = np.asarray(data, dtype=float)
    flag = np.asarray(flag, dtype=int)
    ndata = len(data)

    if np.all(flag == 1):
        return np.percentile(data, np.array(percentiles) * 100)

    data_max = np.max(data)
    data_rs = data_max - data
    order = np.argsort(data_rs)
    x = data_rs[order]
    z = flag[order]
    z[[0, -1]] = [1, 1]

    # survival function
    ss = np.zeros(ndata, dtype=float)
    ss[0] = 1.0
    for i in range(1, ndata):
        r = ndata - i + 1
        if alpha_weight:
            ss[i] = ss[i - 1] * (1 - 1.0 / r) ** z[i - 1] * (1 - ss[i - 1] / r) ** (1 - z[i - 1])
        else:
            ss[i] = ss[i - 1] * (1 - 1.0 / r) ** z[i - 1]

    # CDF: right censored to left censored
    cdf_vals = 1 - ss

    results = []
    for p in percentiles:
        target = 1 - p
        idx = np.searchsorted(cdf_vals, target, side="left")
        idx = np.clip(idx, 0, ndata - 1)
        val = data_max - x[idx]
        results.append(val)

    return np.array(results)