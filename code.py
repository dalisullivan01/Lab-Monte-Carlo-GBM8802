import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================================================
# STEP 1 - LOAD CSV FILES
# =========================================================
# Replace the paths below with your real files.
# Expected format: first column = wavelength (nm), second column = value
# If your CSVs have headers, keep them and use the column names in load_spectrum_csv().

PATH_HBO2 = "HbO2.csv"
PATH_HHB  = "HHb.csv"
PATH_WATER = "water.csv"

def load_spectrum_csv(path, x_col=None, y_col=None):
    """
    Load a CSV containing wavelength-dependent data.
    If x_col and y_col are None, assumes first 2 columns are x and y.
    Returns numpy arrays sorted by wavelength.
    """
    df = pd.read_csv(path)

    if x_col is None or y_col is None:
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
    else:
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)

    order = np.argsort(x)
    return x[order], y[order]


# =========================================================
# STEP 2 - SNAP / RESAMPLE DATA TO A COMMON 2 nm GRID
# =========================================================
# You said you want data points easier to compare to website data.
# This version uses nearest-neighbor assignment onto the common wavelength grid.

WL_MIN = 600
WL_MAX = 1064
WL_STEP = 2
wavelengths_nm = np.arange(WL_MIN, WL_MAX + WL_STEP, WL_STEP)

def snap_to_grid_nearest(x_raw, y_raw, x_grid, max_dist_nm=1.0):
    """
    For each x_grid value, pick the closest raw point if it is close enough.
    Otherwise fill NaN.
    """
    y_grid = np.full_like(x_grid, np.nan, dtype=float)

    for i, xg in enumerate(x_grid):
        idx = np.argmin(np.abs(x_raw - xg))
        if np.abs(x_raw[idx] - xg) <= max_dist_nm:
            y_grid[i] = y_raw[idx]

    return y_grid


# =========================================================
# STEP 3 - INITIALISE ALL VARIABLES
# =========================================================

# ---------- Skin geometry ----------
# Thicknesses from the lab handout, in mm.
layer_names = [
    "stratum_corneum",
    "epiderme",
    "papille_dermique",
    "derme_superieur",
    "derme_reticulaire",
    "derme_profond"
]

layer_thickness_mm = np.array([0.02, 0.25, 0.10, 0.08, 0.20, 0.30], dtype=float)

# Cumulative z boundaries
# Example:
# z = 0.00 to 0.02  -> stratum corneum
# z = 0.02 to 0.27  -> epiderme
# z = 0.27 to 0.37  -> papille dermique
layer_z_edges_mm = np.concatenate(([0.0], np.cumsum(layer_thickness_mm)))

# Boundary where special reflection/transmission must be applied:
# between epiderme and papille dermique
Z_BOUNDARY_EPI_PAP_MM = layer_z_edges_mm[2]   # 0.27 mm

# ---------- Refractive indices ----------
# First 2 layers grouped, last 4 grouped
n_layers = np.array([1.42, 1.42, 1.39, 1.39, 1.39, 1.39], dtype=float)

# ---------- Blood / water fractions ----------
# Vb and Vw from the lab handout
Vb_layers = np.array([0.00, 0.00, 0.04, 0.30, 0.04, 0.10], dtype=float)
Vw_layers = np.array([0.05, 0.20, 0.50, 0.60, 0.70, 0.70], dtype=float)

# Melanin fraction for stratum corneum + epidermis model
Vmel = 0.10

# ---------- Optical constants ----------
g = 0.9
mu_s = 20.1   # mm^-1, constant across layers per lab handout

# ---------- Saturation ----------
SaO2 = 0.98
SvO2 = SaO2 - 0.10   # as specified in the lab handout
# If you want clipping:
SvO2 = max(0.0, min(1.0, SvO2))

# ---------- Mesh ----------
Lx = 10.0    # mm
Lz = 0.95    # mm

Nx = 500     # user can change
Nz = 475     # user can change

dx = Lx / Nx
dz = Lz / Nz

# 2D absorption map A[z, x]
# I store it as [Nz, Nx] so plotting is simpler later.

# ---------- Monte Carlo parameters ----------
Nphotons = 100000   # TODO: choose real value
W0 = 1.0
Wc = 0.01
m = 10

rng = np.random.default_rng(seed=12345)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_layer_index(z_mm):
    """
    Returns the layer index for a given z position in mm.
    Returns None if outside the skin stack.
    """
    if z_mm < 0 or z_mm >= layer_z_edges_mm[-1]:
        return None

    for i in range(len(layer_thickness_mm)):
        if layer_z_edges_mm[i] <= z_mm < layer_z_edges_mm[i + 1]:
            return i

    return None


def mu_a_baseline(lambda_nm):
    """
    Baseline absorption from lab handout:
    mu_a_baseline(lambda) = 7.84e7 * lambda^(-3.255)

    IMPORTANT:
    Make sure lambda is in the unit expected by your formula.
    If your formula expects nm, leave as is.
    If it expects something else, convert here.
    """
    # TODO: verify unit consistency
    return 7.84e7 * (lambda_nm ** (-3.255))


def mu_a_melanin(lambda_nm):
    """
    Melanin absorption from lab handout:
    mu_mel = 6.6e10 * lambda^(-3.33)

    IMPORTANT:
    Same warning about units.
    """
    # TODO: verify unit consistency
    return 6.6e10 * (lambda_nm ** (-3.33))


def build_lookup_dict(x_grid, y_grid):
    """
    Build exact wavelength lookup dictionary after snapping to common grid.
    """
    out = {}
    for x, y in zip(x_grid, y_grid):
        out[float(x)] = float(y) if not np.isnan(y) else np.nan
    return out


def get_value_from_lookup(lookup, lambda_nm):
    """
    Read wavelength value from lookup dict.
    Assumes lambda_nm is exactly on the 2 nm grid.
    """
    val = lookup.get(float(lambda_nm), np.nan)
    if np.isnan(val):
        raise ValueError(f"No spectral data found for wavelength {lambda_nm} nm")
    return val


def compute_mu_a_blood_layer(lambda_nm, layer_idx, hbo2_lookup, hhb_lookup, water_lookup):
    """
    Absorption model for dermal layers.

    Lab says:
    mu_ai = VA_i * mu_A + VV_i * mu_V + Vw_i * mu_w + (1 - (VA_i + VV_i + Vw_i)) * mu_baseline

    with blood ratio artery:vein = 1:1

    You said you want to fill in the exact equations yourself, so the structure is here.
    """
    Vb = Vb_layers[layer_idx]
    Vw = Vw_layers[layer_idx]

    # 1:1 artery-vein ratio
    VA = 0.5 * Vb
    VV = 0.5 * Vb

    mu_hbo2 = get_value_from_lookup(hbo2_lookup, lambda_nm)
    mu_hhb  = get_value_from_lookup(hhb_lookup, lambda_nm)
    mu_w    = get_value_from_lookup(water_lookup, lambda_nm)
    mu_base = mu_a_baseline(lambda_nm)

    # Arterial / venous blood absorption
    mu_A = SaO2 * mu_hbo2 + (1.0 - SaO2) * mu_hhb
    mu_V = SvO2 * mu_hbo2 + (1.0 - SvO2) * mu_hhb

    # TODO:
    # If your literature data are not yet in mm^-1, convert them HERE before combining.
    # Example: if website gives cm^-1, divide by 10 to get mm^-1.
    # mu_hbo2 = ...
    # mu_hhb  = ...
    # mu_w    = ...

    mu_a = VA * mu_A + VV * mu_V + Vw * mu_w + (1.0 - (VA + VV + Vw)) * mu_base
    return mu_a


def compute_mu_a_epi_layer(lambda_nm, layer_idx, water_lookup):
    """
    Absorption model for stratum corneum + epidermis:
    mu_a_epi = Vmel * mu_mel + Vw * mu_w + (1 - (Vmel + Vw)) * mu_baseline
    """
    Vw = Vw_layers[layer_idx]

    mu_w = get_value_from_lookup(water_lookup, lambda_nm)
    mu_mel = mu_a_melanin(lambda_nm)
    mu_base = mu_a_baseline(lambda_nm)

    # TODO:
    # Convert units if needed here too.
    # mu_w = ...

    mu_a = Vmel * mu_mel + Vw * mu_w + (1.0 - (Vmel + Vw)) * mu_base
    return mu_a


def compute_mu_a_layers_for_lambda(lambda_nm, hbo2_lookup, hhb_lookup, water_lookup):
    """
    Returns mu_a for all 6 layers at one wavelength.
    """
    mu_a_layers = np.zeros(6, dtype=float)

    for i in range(6):
        if i in [0, 1]:
            mu_a_layers[i] = compute_mu_a_epi_layer(lambda_nm, i, water_lookup)
        else:
            mu_a_layers[i] = compute_mu_a_blood_layer(lambda_nm, i, hbo2_lookup, hhb_lookup, water_lookup)

    return mu_a_layers


def sample_scattering_angle_HG(g, rng):
    """
    Sample theta from Henyey-Greenstein in a 2D simulation framework.
    Here I sample theta, then later rotate the current 2D direction by +/- theta.

    If you want the exact 3D HG implementation projected into x-z,
    tell me and I will swap this section.
    """
    xi = rng.uniform()

    if np.isclose(g, 0.0):
        cos_theta = 2.0 * xi - 1.0
    else:
        cos_theta = (1.0 / (2.0 * g)) * (
            1.0 + g**2 - ((1.0 - g**2) / (1.0 - g + 2.0 * g * xi))**2
        )

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta


def scatter_direction_2D(ux, uz, g, rng):
    """
    2D scattering update.
    Rotates current direction in x-z plane by +/- theta.
    """
    theta = sample_scattering_angle_HG(g, rng)
    sign = -1.0 if rng.uniform() < 0.5 else 1.0

    current_angle = np.arctan2(ux, uz)   # angle measured from +z axis
    new_angle = current_angle + sign * theta

    ux_new = np.sin(new_angle)
    uz_new = np.cos(new_angle)

    # normalize to avoid numeric drift
    norm = np.sqrt(ux_new**2 + uz_new**2)
    ux_new /= norm
    uz_new /= norm

    return ux_new, uz_new


def crosses_boundary(z_old, z_new, z_boundary):
    """
    True if the segment from z_old to z_new crosses z_boundary.
    """
    return (z_old - z_boundary) * (z_new - z_boundary) < 0.0


def fresnel_reflectance_normalized(cos_theta1, n1, n2):
    """
    Placeholder Fresnel reflectance.
    Your pseudo-code gave:
        r = (n1 cos(theta1) - n2 cos(theta2)) / (n1 cos(theta1) + n2 cos(theta2))

    Usually reflectance uses R = r^2.
    Since your pseudo-code writes RAND < r, I am leaving THIS AS A PLACEHOLDER
    for you to finalize exactly how your lab wants it used.
    """
    # TODO:
    # 1) compute theta2 from Snell
    # 2) compute r
    # 3) decide whether your simulation uses r or R = r^2
    raise NotImplementedError("Fill in Fresnel reflectance/transmission model here.")


def handle_epi_pap_boundary(
    x, z, ux, uz, W, remaining_s, n1, n2, rng
):
    """
    Handle the special boundary step between epidermis and papillary dermis only.

    This function is intentionally left partly blank because you said
    you know the exact equations / conventions to use.

    Expected return:
        x, z, ux, uz, W, remaining_s_after_boundary
    """
    # TODO:
    # 1) move photon exactly to boundary
    # 2) compute incidence angle from uz
    # 3) compute reflected vs transmitted event
    # 4) if reflection:
    #       - update weight with your formula
    #       - flip uz
    # 5) else transmission:
    #       - update weight with your formula
    #       - update direction using Snell
    # 6) continue with remaining distance if you want step splitting
    raise NotImplementedError("Fill in interface handling here.")


@dataclass
class PhotonResult:
    max_z_reached_mm: float
    terminated_normally: bool


def simulate_one_wavelength(lambda_nm, hbo2_lookup, hhb_lookup, water_lookup):
    """
    Runs the Monte Carlo simulation for ONE wavelength.
    Returns:
        absorption_map      -> 2D array [Nz, Nx]
        max_z_global_mm     -> deepest z reached by any photon
        n_photons_at_max_z  -> number of photons reaching that same deepest z
    """
    A = np.zeros((Nz, Nx), dtype=float)

    mu_a_layers = compute_mu_a_layers_for_lambda(lambda_nm, hbo2_lookup, hhb_lookup, water_lookup)

    max_z_global_mm = 0.0
    n_photons_at_max_z = 0
    tol_depth_mm = dz / 2.0

    for _ in range(Nphotons):
        # STEP 4 - Create photon packet
        x = Lx / 2.0
        z = 0.0
        ux = 0.0
        uz = 1.0
        W = W0

        photon_max_z = z

        while True:
            layer_idx = get_layer_index(z)

            # Outside model
            if layer_idx is None:
                break

            mu_a = mu_a_layers[layer_idx]
            mu_t = mu_a + mu_s

            if mu_t <= 0:
                break

            # Sample step
            s = -np.log(rng.uniform()) / mu_t

            x_old, z_old = x, z
            x_new = x + ux * s
            z_new = z + uz * s

            # STEP 5 - Boundary crossing only between epidermis and papillary dermis
            if crosses_boundary(z_old, z_new, Z_BOUNDARY_EPI_PAP_MM):
                # Only handle if this is the epi <-> papillary boundary
                # i.e. crossing between layer 1 and layer 2
                idx_old = get_layer_index(z_old + 1e-12 * np.sign(uz))
                idx_new = get_layer_index(z_new - 1e-12 * np.sign(uz))

                if {idx_old, idx_new} == {1, 2}:
                    remaining_s = s

                    # choose correct n1, n2 depending on direction
                    if idx_old == 1 and idx_new == 2:
                        n1, n2 = n_layers[1], n_layers[2]
                    else:
                        n1, n2 = n_layers[2], n_layers[1]

                    # TODO: fill this function with your exact Fresnel / Snell equations
                    x, z, ux, uz, W, remaining_s = handle_epi_pap_boundary(
                        x_old, z_old, ux, uz, W, remaining_s, n1, n2, rng
                    )

                    # If you want to continue the remaining distance after transmission/reflection,
                    # do it inside handle_epi_pap_boundary().
                    # For now we continue to next while iteration.
                    photon_max_z = max(photon_max_z, z)
                    if W <= 0:
                        break
                    continue

            # Normal propagation
            x, z = x_new, z_new

            # Exit mesh if outside x or z range
            if x < 0 or x >= Lx or z < 0 or z >= Lz:
                photon_max_z = max(photon_max_z, min(max(z, 0.0), Lz))
                break

            # Absorption
            deltaW = (mu_a / mu_t) * W
            W -= deltaW

            ix = int(x / dx)
            iz = int(z / dz)

            if 0 <= ix < Nx and 0 <= iz < Nz:
                A[iz, ix] += deltaW

            photon_max_z = max(photon_max_z, z)

            # Roulette
            if W < Wc:
                if rng.uniform() <= 1.0 / m:
                    W *= m
                else:
                    W = 0.0
                    break

            # Scatter
            ux, uz = scatter_direction_2D(ux, uz, g, rng)

        # Track deepest penetration metric
        if photon_max_z > max_z_global_mm + tol_depth_mm:
            max_z_global_mm = photon_max_z
            n_photons_at_max_z = 1
        elif abs(photon_max_z - max_z_global_mm) <= tol_depth_mm:
            n_photons_at_max_z += 1

    A /= Nphotons
    return A, max_z_global_mm, n_photons_at_max_z


# =========================================================
# MAIN SCRIPT
# =========================================================

# Load raw spectra
x_hbo2_raw, y_hbo2_raw = load_spectrum_csv(PATH_HBO2)
x_hhb_raw,  y_hhb_raw  = load_spectrum_csv(PATH_HHB)
x_w_raw,    y_w_raw    = load_spectrum_csv(PATH_WATER)

# Snap all spectra onto the same 2 nm grid
y_hbo2_grid = snap_to_grid_nearest(x_hbo2_raw, y_hbo2_raw, wavelengths_nm, max_dist_nm=1.0)
y_hhb_grid  = snap_to_grid_nearest(x_hhb_raw,  y_hhb_raw,  wavelengths_nm, max_dist_nm=1.0)
y_w_grid    = snap_to_grid_nearest(x_w_raw,    y_w_raw,    wavelengths_nm, max_dist_nm=1.0)

# Build exact lookup dictionaries
hbo2_lookup = build_lookup_dict(wavelengths_nm, y_hbo2_grid)
hhb_lookup  = build_lookup_dict(wavelengths_nm, y_hhb_grid)
water_lookup = build_lookup_dict(wavelengths_nm, y_w_grid)

# Sweep all wavelengths
results = []
best_A = None
best_lambda = None

for lam in wavelengths_nm:
    print(f"Running wavelength = {lam} nm")

    # If your CSVs do not have every 2 nm value after snapping, this will fail here.
    # That is good because it forces you to verify your data coverage.
    A_lam, max_z_lam, n_photons_lam = simulate_one_wavelength(
        lam, hbo2_lookup, hhb_lookup, water_lookup
    )

    results.append({
        "wavelength_nm": lam,
        "max_z_reached_mm": max_z_lam,
        "n_photons_at_max_depth": n_photons_lam
    })

results_df = pd.DataFrame(results)

# Select best wavelength:
# first maximize depth, then maximize number of photons reaching that depth
results_df = results_df.sort_values(
    by=["max_z_reached_mm", "n_photons_at_max_depth"],
    ascending=[False, False]
).reset_index(drop=True)

best_lambda = results_df.loc[0, "wavelength_nm"]

print("\n===== BEST WAVELENGTH RESULT =====")
print(results_df.head(10))
print(f"\nBest wavelength = {best_lambda} nm")

# Re-run best wavelength once more if you want its absorption map for plotting
best_A, best_max_z, best_nphot = simulate_one_wavelength(
    best_lambda, hbo2_lookup, hhb_lookup, water_lookup
)

print(f"Best max depth = {best_max_z:.4f} mm")
print(f"Photons reaching that depth = {best_nphot}")

# Save summary
results_df.to_csv("question1_wavelength_sweep_results.csv", index=False)

# Optional plots
plt.figure(figsize=(7, 4))
plt.plot(results_df["wavelength_nm"], results_df["max_z_reached_mm"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Max z reached (mm)")
plt.title("Penetration depth vs wavelength")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(results_df["wavelength_nm"], results_df["n_photons_at_max_depth"])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Photons reaching max depth")
plt.title("Photon count at max depth vs wavelength")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(8, 4))
plt.imshow(
    np.log10(best_A + 1e-15),
    extent=[0, Lx, Lz, 0],
    aspect='auto',
    cmap='viridis'
)
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.title(f"log10 Absorption map at best wavelength = {best_lambda} nm")
plt.colorbar(label="log10(A)")
plt.show()