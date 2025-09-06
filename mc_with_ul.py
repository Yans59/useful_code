### ---- Shulan Yan 20250831
### This is an example of mcmc fitting with upper limits. 
### More details can be found in python package PyMC webpage.
### Suppose that you have detected value: x1, x2, y_det, y_det_err and upper limits: x1_u, x2_u, y_ul
import numpy as np
import pymc as pm

X_det = np.column_stack([x1, x2])
X_ul = np.column_stack([x1_u, x2_u])

# # ----------------------------

# ---------- STANDARDIZE ----------
X_all = np.vstack([X_det, X_ul])
x_mean = X_all.mean(axis=0)
x_std = X_all.std(axis=0)

X_det_s = (X_det - x_mean) / x_std
X_ul_s = (X_ul - x_mean) / x_std

y_mean = np.mean(y_det)
y_std = np.std(y_det)
y_det_s = (y_det - y_mean) / y_std
y_ul_s = (y_ul_arr - y_mean) / y_std

# upper limits scale：<1 close to upper limits >1 reduce the influence of upper limits
yerr_ul_scale = 1

yerr_ul = yerr_ul_scale * np.median(yerr_det) * np.ones_like(y_ul_s)

yerr_det_s = yerr_det / y_std
yerr_ul_s = yerr_ul / y_std

with pm.Model() as model:
    # 收紧 prior
    m1 = pm.TruncatedNormal("m1", mu=0, sigma=3, lower=-4, upper=1)
    m2 = pm.TruncatedNormal("m2", mu=0, sigma=3, lower=-4, upper=1)
    b  = pm.Normal("b", mu=5, sigma=10)  # 放宽 b 的 prior，增加灵活性

    # 内禀误差
    sigma_int = pm.HalfNormal("sigma_int", sigma=2)  # 增大内禀误差的 prior

    # -------------------
    # 探测点 likelihood (StudentT)
    mu_det = m1 * X_det_s[:, 0] + m2 * X_det_s[:, 1] + b
    sigma_det = pmm.sqrt(sigma_int**2 + yerr_det_s**2)
    pm.StudentT("y_det", nu=4, mu=mu_det, sigma=sigma_det, observed=y_det_s)

    # -------------------
    # 上限点 likelihood (Censored)
    mu_ul_vec = m1 * X_ul_s[:, 0] + m2 * X_ul_s[:, 1] + b
    sigma_ul_vec = pmm.sqrt(sigma_int**2 + yerr_ul_s**2)

    pm.Censored(
        "y_ul_obs",
        pm.Normal.dist(mu=mu_ul_vec, sigma=sigma_ul_vec),
        lower=-np.inf,
        upper=y_ul_s,
        observed=y_ul_s
    )

    # -------------------
    # MAP 初始化
    start = pm.find_MAP()

    # MCMC 采样
    idata = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        target_accept=0.95,
        start=start,
        random_seed=42,
        progressbar=True
    )

# ---------- POSTPROCESS: anti STANDARDIZE to original data----------

m1_ch = idata.posterior["m1"].values        # shape (n_chains, n_draws)
m2_ch = idata.posterior["m2"].values
b_ch  = idata.posterior["b"].values
sigma_int_det_ch = idata.posterior["sigma_int_det"].values

# anti STANDARDIZE
m1_phys = m1_ch * (y_std / x_std[0])
m2_phys = m2_ch * (y_std / x_std[1])
b_phys  = (b_ch * y_std) + y_mean - (m1_phys * x_mean[0]) - (m2_phys * x_mean[1])
sigma_int_det_phys = sigma_int_det_ch * y_std

# make a new idata_phys（used to trace/summary）
idata_phys = az.from_dict(
    posterior={
        "m1": m1_phys,
        "m2": m2_phys,
        "b": b_phys,
        "sigma_int": sigma_int_det_phys
    },
    coords=None,
    dims=None
)

# ---------- ArviZ summary ----------
display(az.summary(idata_phys, var_names=["m1","m2","b","sigma_int"], round_to=3))

### Plot corner figure
m1_flat = m1_phys.reshape(-1)
m2_flat = m2_phys.reshape(-1)
b_flat  = b_phys.reshape(-1)
s_flat  = sigma_int_det_phys.reshape(-1)

param_samples = np.column_stack([m1_flat, m2_flat, b_flat, s_flat])
means = np.mean(param_samples, axis=0) ## If you need median, you can change np.mean to np.median
print('medians -> m1: %.4f, m2: %.4f, b: %.4f, sigma_int: %.4f' % tuple(means))

corner.corner(
    param_samples,
    labels=["m1","m2","b","sigma_int"],
    show_titles=True,
    title_fmt=".3f",
    truths=means
)
axs[0].hist(np.hstack([Def_fh1_obs, Def_fh1_upper]), range = (-1.2, 1), density = True, color = 'gray')

### plot trace
az.plot_trace(idata_phys, var_names=["m1","m2","b"])
plt.subplots_adjust(hspace=0.5)