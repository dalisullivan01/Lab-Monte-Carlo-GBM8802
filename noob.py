import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# STEP 1 - FILE PATHS
# =========================================================

path_hbo2_file = "HbO2.csv"
path_hhb_file = "HHb.csv"
path_water_file = "water.csv"

# =========================================================
# STEP 2 - WAVELENGTH GRID
# =========================================================

minimum_wavelength_nm = 600
maximum_wavelength_nm = 1064
wavelength_step_nm = 2

all_wavelengths_nm = np.arange(
    minimum_wavelength_nm,
    maximum_wavelength_nm + wavelength_step_nm,
    wavelength_step_nm
)

# =========================================================
# STEP 3 - SKIN LAYER INFORMATION
# =========================================================

skin_layer_names = [
    "stratum_corneum",
    "epiderme",
    "papille_dermique",
    "derme_superieur",
    "derme_reticulaire",
    "derme_profond"
]

skin_layer_thicknesses_mm = np.array([0.02, 0.25, 0.10, 0.08, 0.20, 0.30], dtype=float)

skin_layer_z_boundaries_mm = [0.0]

current_total_depth = 0.0
for thickness_value in skin_layer_thicknesses_mm:
    current_total_depth = current_total_depth + thickness_value
    skin_layer_z_boundaries_mm.append(current_total_depth)

skin_layer_z_boundaries_mm = np.array(skin_layer_z_boundaries_mm, dtype=float)

# Boundary between epidermis and papillary dermis
z_boundary_epidermis_to_papillary_mm = skin_layer_z_boundaries_mm[2]

# =========================================================
# STEP 4 - REFRACTIVE INDICES
# =========================================================

# First 2 layers: 1.42
# Last 4 layers: 1.39
refractive_index_per_layer = np.array([1.42, 1.42, 1.39, 1.39, 1.39, 1.39], dtype=float)

# =========================================================
# STEP 5 - BLOOD AND WATER FRACTIONS
# =========================================================

blood_volume_fraction_per_layer = np.array([0.00, 0.00, 0.04, 0.30, 0.04, 0.10], dtype=float)
water_volume_fraction_per_layer = np.array([0.05, 0.20, 0.50, 0.60, 0.70, 0.70], dtype=float)

melanin_volume_fraction = 0.10

# =========================================================
# STEP 6 - OPTICAL PARAMETERS
# =========================================================

anisotropy_g = 0.9
scattering_coefficient_mu_s = 20.1   # mm^-1

arterial_oxygen_saturation = 0.98
venous_oxygen_saturation = 0.9 * arterial_oxygen_saturation

if venous_oxygen_saturation < 0.0:
    venous_oxygen_saturation = 0.0

if venous_oxygen_saturation > 1.0:
    venous_oxygen_saturation = 1.0

# =========================================================
# STEP 7 - MESH
# =========================================================

simulation_width_x_mm = 10.0
simulation_depth_z_mm = 0.95

number_of_x_cells = 500
number_of_z_cells = 475

cell_size_x_mm = simulation_width_x_mm / number_of_x_cells
cell_size_z_mm = simulation_depth_z_mm / number_of_z_cells

# =========================================================
# STEP 8 - MONTE CARLO PARAMETERS
# =========================================================

number_of_photons = 100000
initial_photon_weight = 1.0
roulette_weight_threshold = 0.01
roulette_multiplier = 10

random_number_generator = np.random.default_rng(seed=12345)

# =========================================================
# HELPER FUNCTION 1 - LOAD CSV
# =========================================================

def load_spectrum_csv(file_path, x_column_name=None, y_column_name=None):
    """
    Load one CSV file.
    If no column names are given, it assumes:
    first column = x
    second column = y
    Then it sorts the result by x value.
    """

    data_frame = pd.read_csv(file_path)

    if x_column_name is None or y_column_name is None:
        x_values = data_frame.iloc[:, 0].to_numpy(dtype=float)
        y_values = data_frame.iloc[:, 1].to_numpy(dtype=float)
    else:
        x_values = data_frame[x_column_name].to_numpy(dtype=float)
        y_values = data_frame[y_column_name].to_numpy(dtype=float)

    sort_order = np.argsort(x_values)

    sorted_x_values = x_values[sort_order]
    sorted_y_values = y_values[sort_order]

    return sorted_x_values, sorted_y_values

# =========================================================
# HELPER FUNCTION 2 - SNAP DATA TO COMMON GRID
# =========================================================

def snap_raw_data_to_common_grid_nearest(raw_x_values, raw_y_values, target_x_grid, maximum_allowed_distance_nm=1.0):
    """
    For each wavelength in target_x_grid:
    - find the closest wavelength in raw_x_values
    - if it is close enough, use that y value
    - if not, store NaN
    """

    snapped_y_values = np.full(len(target_x_grid), np.nan, dtype=float)

    for grid_index in range(len(target_x_grid)):
        target_wavelength = target_x_grid[grid_index]

        distance_array = np.abs(raw_x_values - target_wavelength)
        closest_point_index = np.argmin(distance_array)
        closest_distance = distance_array[closest_point_index]

        if closest_distance <= maximum_allowed_distance_nm:
            snapped_y_values[grid_index] = raw_y_values[closest_point_index]

    return snapped_y_values

# =========================================================
# HELPER FUNCTION 3 - BUILD LOOKUP DICTIONARY
# =========================================================

def build_wavelength_lookup_dictionary(wavelength_array, value_array):
    """
    Build a dictionary like:
    lookup[600.0] = value_at_600
    lookup[602.0] = value_at_602
    """

    wavelength_lookup_dictionary = {}

    for i in range(len(wavelength_array)):
        current_wavelength = float(wavelength_array[i])
        current_value = value_array[i]

        if np.isnan(current_value):
            wavelength_lookup_dictionary[current_wavelength] = np.nan
        else:
            wavelength_lookup_dictionary[current_wavelength] = float(current_value)

    return wavelength_lookup_dictionary

# =========================================================
# HELPER FUNCTION 4 - GET VALUE FROM LOOKUP
# =========================================================

def get_value_from_lookup_dictionary(wavelength_lookup_dictionary, wavelength_nm):
    """
    Read one value from the lookup dictionary.
    """

    wavelength_key = float(wavelength_nm)

    if wavelength_key not in wavelength_lookup_dictionary:
        raise ValueError("Wavelength not found in lookup dictionary: " + str(wavelength_nm))

    value = wavelength_lookup_dictionary[wavelength_key]

    if np.isnan(value):
        raise ValueError("Wavelength exists but value is NaN: " + str(wavelength_nm))

    return value

# =========================================================
# HELPER FUNCTION 5 - GET LAYER INDEX FROM Z POSITION
# =========================================================

def get_skin_layer_index_from_z_position(z_position_mm):
    """
    Returns:
    0 for layer 1
    1 for layer 2
    ...
    5 for layer 6
    Returns None if z is outside all skin layers.
    """

    if z_position_mm < 0:
        return None

    if z_position_mm >= skin_layer_z_boundaries_mm[-1]:
        return None

    for layer_index in range(len(skin_layer_thicknesses_mm)):
        lower_boundary = skin_layer_z_boundaries_mm[layer_index]
        upper_boundary = skin_layer_z_boundaries_mm[layer_index + 1]

        if lower_boundary <= z_position_mm < upper_boundary:
            return layer_index

    return None

# =========================================================
# HELPER FUNCTION 6 - BASELINE ABSORPTION
# =========================================================

def compute_baseline_absorption_coefficient(lambda_nm):
    """
    From the notebook:
    mu_a_baseline(lambda) = 7.84e7 * lambda^(-3.255)

    IMPORTANT:
    Check wavelength units.
    """
    baseline_absorption_value = 7.84e7 * (lambda_nm ** (-3.255))
    return baseline_absorption_value

# =========================================================
# HELPER FUNCTION 7 - MELANIN ABSORPTION
# =========================================================

def compute_melanin_absorption_coefficient(lambda_nm):
    """
    From the notebook:
    mu_a_mel(lambda) = 6.6e10 * lambda^(-3.33)

    IMPORTANT:
    Check wavelength units.
    """
    melanin_absorption_value = 6.6e10 * (lambda_nm ** (-3.33))
    return melanin_absorption_value

# =========================================================
# HELPER FUNCTION 8 - BLOOD LAYER ABSORPTION
# =========================================================

def compute_absorption_for_blood_based_layer(
    lambda_nm,
    skin_layer_index,
    hbo2_lookup_dictionary,
    hhb_lookup_dictionary,
    water_lookup_dictionary
):
    """
    Absorption coefficient for the last 4 skin layers.

    Notebook formulas used:

    mu_A(lambda) = 0.98 * mu_a_HbO2(lambda) + 0.02 * mu_a_HHb(lambda)

    mu_ai(lambda) =
        1.9 * V_Ai * mu_A(lambda)
        + 0.1 * mu_a_HHb(lambda)
        + V_wi * mu_wi(lambda)
        + (1 - (2 * V_Ai + V_wi)) * mu_a_baseline(lambda)
    """

    total_blood_fraction = blood_volume_fraction_per_layer[skin_layer_index]
    water_fraction = water_volume_fraction_per_layer[skin_layer_index]

    # In your derivation, V_A = V_V and total blood fraction is split equally
    arterial_fraction = total_blood_fraction / 2.0

    mu_a_hbo2_value = get_value_from_lookup_dictionary(hbo2_lookup_dictionary, lambda_nm)
    mu_a_hhb_value = get_value_from_lookup_dictionary(hhb_lookup_dictionary, lambda_nm)
    mu_a_water_value = get_value_from_lookup_dictionary(water_lookup_dictionary, lambda_nm)
    mu_a_baseline_value = compute_baseline_absorption_coefficient(lambda_nm)

    # =====================================================
    # TODO: UNIT CONVERSION SECTION
    # Put every needed unit conversion here.
    #
    # Example only:
    # mu_a_water_value = mu_a_water_value / 10.0
    #
    # For HbO2 / HHb, if you are using OMLC raw data directly,
    # you may still need a conversion depending on your chosen formula.
    # =====================================================

    arterial_blood_absorption = (
        0.98 * mu_a_hbo2_value
        + 0.02 * mu_a_hhb_value
    )

    blood_absorption_term = 1.9 * arterial_fraction * arterial_blood_absorption

    extra_hhb_term = 0.1 * mu_a_hhb_value

    water_absorption_term = water_fraction * mu_a_water_value

    baseline_fraction = 1.0 - (2.0 * arterial_fraction + water_fraction)
    baseline_absorption_term = baseline_fraction * mu_a_baseline_value

    total_absorption = (
        blood_absorption_term
        + extra_hhb_term
        + water_absorption_term
        + baseline_absorption_term
    )

    return total_absorption

# =========================================================
# HELPER FUNCTION 9 - EPIDERMIS / STRATUM CORNEUM ABSORPTION
# =========================================================

def compute_absorption_for_epidermis_type_layer(
    lambda_nm,
    skin_layer_index,
    water_lookup_dictionary
):
    """
    Absorption coefficient for the first 2 layers.

    Notebook formula used:

    mu_a_epi(lambda) =
        0.1 * mu_a_mel(lambda)
        + V_w_epi * mu_w(lambda)
        + (0.9 - V_w_epi) * mu_a_baseline(lambda)
    """

    water_fraction = water_volume_fraction_per_layer[skin_layer_index]

    mu_a_water_value = get_value_from_lookup_dictionary(water_lookup_dictionary, lambda_nm)
    mu_a_melanin_value = compute_melanin_absorption_coefficient(lambda_nm)
    mu_a_baseline_value = compute_baseline_absorption_coefficient(lambda_nm)

    # =====================================================
    # TODO: UNIT CONVERSION SECTION
    # Put water conversion here if needed.
    # Example:
    # mu_a_water_value = mu_a_water_value / 10.0
    # =====================================================

    melanin_absorption_term = 0.1 * mu_a_melanin_value
    water_absorption_term = water_fraction * mu_a_water_value
    baseline_absorption_term = (0.9 - water_fraction) * mu_a_baseline_value

    total_absorption = (
        melanin_absorption_term
        + water_absorption_term
        + baseline_absorption_term
    )

    return total_absorption

# =========================================================
# HELPER FUNCTION 10 - ALL 6 ABSORPTION COEFFICIENTS
# =========================================================

def compute_all_layer_absorptions_for_one_wavelength(
    lambda_nm,
    hbo2_lookup_dictionary,
    hhb_lookup_dictionary,
    water_lookup_dictionary
):
    """
    Returns one array with the 6 absorption coefficients,
    one for each skin layer.
    """

    all_layer_absorption_values = np.zeros(6, dtype=float)

    for skin_layer_index in range(6):

        if skin_layer_index == 0 or skin_layer_index == 1:
            current_absorption_value = compute_absorption_for_epidermis_type_layer(
                lambda_nm,
                skin_layer_index,
                water_lookup_dictionary
            )
        else:
            current_absorption_value = compute_absorption_for_blood_based_layer(
                lambda_nm,
                skin_layer_index,
                hbo2_lookup_dictionary,
                hhb_lookup_dictionary,
                water_lookup_dictionary
            )

        all_layer_absorption_values[skin_layer_index] = current_absorption_value

    return all_layer_absorption_values

# =========================================================
# HELPER FUNCTION 11 - SAMPLE HG SCATTERING ANGLE
# =========================================================

def sample_henyey_greenstein_scattering_angle(g_value, rng_object):
    """
    Sample one scattering angle from the HG distribution.
    """

    random_number = rng_object.uniform()

    if np.isclose(g_value, 0.0):
        cosine_theta = 2.0 * random_number - 1.0
    else:
        numerator_part_1 = 1.0 + g_value**2
        numerator_part_2 = (1.0 - g_value**2)
        denominator_part = (1.0 - g_value + 2.0 * g_value * random_number)

        fraction_term = numerator_part_2 / denominator_part
        fraction_term_squared = fraction_term**2

        cosine_theta = (1.0 / (2.0 * g_value)) * (numerator_part_1 - fraction_term_squared)

    if cosine_theta < -1.0:
        cosine_theta = -1.0
    if cosine_theta > 1.0:
        cosine_theta = 1.0

    theta_radians = np.arccos(cosine_theta)

    return theta_radians

# =========================================================
# HELPER FUNCTION 12 - UPDATE 2D DIRECTION AFTER SCATTER
# =========================================================

def update_direction_after_2d_scattering(old_ux, old_uz, g_value, rng_object):
    """
    Rotate the x-z direction by a random scattering angle.
    """

    scattering_angle = sample_henyey_greenstein_scattering_angle(g_value, rng_object)

    random_sign_test = rng_object.uniform()

    if random_sign_test < 0.5:
        scattering_sign = -1.0
    else:
        scattering_sign = 1.0

    old_angle_from_z_axis = np.arctan2(old_ux, old_uz)
    new_angle_from_z_axis = old_angle_from_z_axis + scattering_sign * scattering_angle

    new_ux = np.sin(new_angle_from_z_axis)
    new_uz = np.cos(new_angle_from_z_axis)

    direction_norm = np.sqrt(new_ux**2 + new_uz**2)

    new_ux = new_ux / direction_norm
    new_uz = new_uz / direction_norm

    return new_ux, new_uz

# =========================================================
# HELPER FUNCTION 13 - CHECK IF BOUNDARY WAS CROSSED
# =========================================================

def check_if_step_crosses_boundary(old_z_position, new_z_position, boundary_z_position):
    """
    Returns True if the photon crossed the given z boundary.
    """

    left_side = old_z_position - boundary_z_position
    right_side = new_z_position - boundary_z_position

    crossed_boundary = (left_side * right_side) < 0.0
    return crossed_boundary

# =========================================================
# HELPER FUNCTION 14 - FRESNEL PLACEHOLDER
# =========================================================

def compute_fresnel_reflectance_placeholder(cos_theta_1, refractive_index_1, refractive_index_2):
    """
    Placeholder for Fresnel reflectance.

    You still need to decide:
    - exact Snell implementation
    - whether you compare rand < r or rand < R
    """

    raise NotImplementedError("You still need to fill the Fresnel reflectance code here.")

# =========================================================
# HELPER FUNCTION 15 - SPECIAL BOUNDARY HANDLING PLACEHOLDER
# =========================================================

def handle_epidermis_papillary_boundary(
    old_x_position,
    old_z_position,
    old_ux,
    old_uz,
    old_weight,
    remaining_step_length,
    refractive_index_1,
    refractive_index_2,
    rng_object
):
    """
    Placeholder for the special reflection / transmission rule
    at the epidermis / papillary dermis boundary.
    """

    raise NotImplementedError("You still need to fill the special boundary code here.")

# =========================================================
# HELPER FUNCTION 16 - SIMULATE ONE WAVELENGTH
# =========================================================

def simulate_one_wavelength(
    lambda_nm,
    hbo2_lookup_dictionary,
    hhb_lookup_dictionary,
    water_lookup_dictionary
):
    """
    Runs the full Monte Carlo simulation for one wavelength.
    Returns:
    - absorption map
    - max z reached
    - number of photons that reached that same max z
    """

    absorption_map = np.zeros((number_of_z_cells, number_of_x_cells), dtype=float)

    layer_absorption_values = compute_all_layer_absorptions_for_one_wavelength(
        lambda_nm,
        hbo2_lookup_dictionary,
        hhb_lookup_dictionary,
        water_lookup_dictionary
    )

    global_maximum_z_reached_mm = 0.0
    number_of_photons_at_maximum_z = 0
    depth_tolerance_mm = cell_size_z_mm / 2.0

    for photon_index in range(number_of_photons):

        photon_x_position_mm = simulation_width_x_mm / 2.0
        photon_z_position_mm = 0.0

        photon_ux = 0.0
        photon_uz = 1.0

        photon_weight = initial_photon_weight

        photon_maximum_z_reached_mm = photon_z_position_mm

        while True:

            current_skin_layer_index = get_skin_layer_index_from_z_position(photon_z_position_mm)

            if current_skin_layer_index is None:
                break

            current_absorption_coefficient = layer_absorption_values[current_skin_layer_index]
            current_total_interaction_coefficient = current_absorption_coefficient + scattering_coefficient_mu_s

            if current_total_interaction_coefficient <= 0:
                break

            random_number_for_step = random_number_generator.uniform()
            step_length_mm = -np.log(random_number_for_step) / current_total_interaction_coefficient

            old_x_position_mm = photon_x_position_mm
            old_z_position_mm = photon_z_position_mm

            new_x_position_mm = old_x_position_mm + photon_ux * step_length_mm
            new_z_position_mm = old_z_position_mm + photon_uz * step_length_mm

            crossed_special_boundary = check_if_step_crosses_boundary(
                old_z_position_mm,
                new_z_position_mm,
                z_boundary_epidermis_to_papillary_mm
            )

            if crossed_special_boundary:

                old_side_layer_index = get_skin_layer_index_from_z_position(
                    old_z_position_mm + 1e-12 * np.sign(photon_uz)
                )
                new_side_layer_index = get_skin_layer_index_from_z_position(
                    new_z_position_mm - 1e-12 * np.sign(photon_uz)
                )

                if {old_side_layer_index, new_side_layer_index} == {1, 2}:

                    remaining_step_after_boundary = step_length_mm

                    if old_side_layer_index == 1 and new_side_layer_index == 2:
                        refractive_index_before_boundary = refractive_index_per_layer[1]
                        refractive_index_after_boundary = refractive_index_per_layer[2]
                    else:
                        refractive_index_before_boundary = refractive_index_per_layer[2]
                        refractive_index_after_boundary = refractive_index_per_layer[1]

                    photon_x_position_mm, photon_z_position_mm, photon_ux, photon_uz, photon_weight, remaining_step_after_boundary = handle_epidermis_papillary_boundary(
                        old_x_position_mm,
                        old_z_position_mm,
                        photon_ux,
                        photon_uz,
                        photon_weight,
                        remaining_step_after_boundary,
                        refractive_index_before_boundary,
                        refractive_index_after_boundary,
                        random_number_generator
                    )

                    if photon_z_position_mm > photon_maximum_z_reached_mm:
                        photon_maximum_z_reached_mm = photon_z_position_mm

                    if photon_weight <= 0:
                        break

                    continue

            photon_x_position_mm = new_x_position_mm
            photon_z_position_mm = new_z_position_mm

            if photon_x_position_mm < 0 or photon_x_position_mm >= simulation_width_x_mm:
                if photon_z_position_mm > photon_maximum_z_reached_mm:
                    photon_maximum_z_reached_mm = photon_z_position_mm
                break

            if photon_z_position_mm < 0 or photon_z_position_mm >= simulation_depth_z_mm:
                clipped_z_position = min(max(photon_z_position_mm, 0.0), simulation_depth_z_mm)
                if clipped_z_position > photon_maximum_z_reached_mm:
                    photon_maximum_z_reached_mm = clipped_z_position
                break

            absorbed_weight = (current_absorption_coefficient / current_total_interaction_coefficient) * photon_weight
            photon_weight = photon_weight - absorbed_weight

            x_cell_index = int(photon_x_position_mm / cell_size_x_mm)
            z_cell_index = int(photon_z_position_mm / cell_size_z_mm)

            if 0 <= x_cell_index < number_of_x_cells and 0 <= z_cell_index < number_of_z_cells:
                absorption_map[z_cell_index, x_cell_index] = absorption_map[z_cell_index, x_cell_index] + absorbed_weight

            if photon_z_position_mm > photon_maximum_z_reached_mm:
                photon_maximum_z_reached_mm = photon_z_position_mm

            if photon_weight < roulette_weight_threshold:
                roulette_random_number = random_number_generator.uniform()

                if roulette_random_number <= (1.0 / roulette_multiplier):
                    photon_weight = photon_weight * roulette_multiplier
                else:
                    photon_weight = 0.0
                    break

            photon_ux, photon_uz = update_direction_after_2d_scattering(
                photon_ux,
                photon_uz,
                anisotropy_g,
                random_number_generator
            )

        if photon_maximum_z_reached_mm > global_maximum_z_reached_mm + depth_tolerance_mm:
            global_maximum_z_reached_mm = photon_maximum_z_reached_mm
            number_of_photons_at_maximum_z = 1

        elif abs(photon_maximum_z_reached_mm - global_maximum_z_reached_mm) <= depth_tolerance_mm:
            number_of_photons_at_maximum_z = number_of_photons_at_maximum_z + 1

    absorption_map = absorption_map / number_of_photons

    return absorption_map, global_maximum_z_reached_mm, number_of_photons_at_maximum_z

# =========================================================
# MAIN PART OF PROGRAM
# =========================================================

def main():
    raw_hbo2_wavelengths_nm, raw_hbo2_values = load_spectrum_csv(path_hbo2_file)
    raw_hhb_wavelengths_nm, raw_hhb_values = load_spectrum_csv(path_hhb_file)
    raw_water_wavelengths_nm, raw_water_values = load_spectrum_csv(path_water_file)

    grid_hbo2_values = snap_raw_data_to_common_grid_nearest(
        raw_hbo2_wavelengths_nm,
        raw_hbo2_values,
        all_wavelengths_nm,
        maximum_allowed_distance_nm=1.0
    )

    grid_hhb_values = snap_raw_data_to_common_grid_nearest(
        raw_hhb_wavelengths_nm,
        raw_hhb_values,
        all_wavelengths_nm,
        maximum_allowed_distance_nm=1.0
    )

    grid_water_values = snap_raw_data_to_common_grid_nearest(
        raw_water_wavelengths_nm,
        raw_water_values,
        all_wavelengths_nm,
        maximum_allowed_distance_nm=1.0
    )

    hbo2_lookup_dictionary = build_wavelength_lookup_dictionary(all_wavelengths_nm, grid_hbo2_values)
    hhb_lookup_dictionary = build_wavelength_lookup_dictionary(all_wavelengths_nm, grid_hhb_values)
    water_lookup_dictionary = build_wavelength_lookup_dictionary(all_wavelengths_nm, grid_water_values)

    all_results = []

    for current_wavelength_nm in all_wavelengths_nm:
        print("Running wavelength =", current_wavelength_nm, "nm")

        current_absorption_map, current_maximum_z_mm, current_number_of_photons_at_maximum_z = simulate_one_wavelength(
            current_wavelength_nm,
            hbo2_lookup_dictionary,
            hhb_lookup_dictionary,
            water_lookup_dictionary
        )

        one_result_dictionary = {
            "wavelength_nm": current_wavelength_nm,
            "max_z_reached_mm": current_maximum_z_mm,
            "n_photons_at_max_depth": current_number_of_photons_at_maximum_z
        }

        all_results.append(one_result_dictionary)

    results_data_frame = pd.DataFrame(all_results)

    results_data_frame = results_data_frame.sort_values(
        by=["max_z_reached_mm", "n_photons_at_max_depth"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best_wavelength_nm = results_data_frame.loc[0, "wavelength_nm"]

    print()
    print("===== BEST WAVELENGTH RESULT =====")
    print(results_data_frame.head(10))
    print()
    print("Best wavelength =", best_wavelength_nm, "nm")

    best_absorption_map, best_maximum_z_mm, best_number_of_photons_at_maximum_z = simulate_one_wavelength(
        best_wavelength_nm,
        hbo2_lookup_dictionary,
        hhb_lookup_dictionary,
        water_lookup_dictionary
    )

    print("Best max depth =", round(best_maximum_z_mm, 4), "mm")
    print("Photons reaching that depth =", best_number_of_photons_at_maximum_z)

    results_data_frame.to_csv("question1_wavelength_sweep_results.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(results_data_frame["wavelength_nm"], results_data_frame["max_z_reached_mm"])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Max z reached (mm)")
    plt.title("Penetration depth vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(results_data_frame["wavelength_nm"], results_data_frame["n_photons_at_max_depth"])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Photons reaching max depth")
    plt.title("Photon count at max depth vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.imshow(
        np.log10(best_absorption_map + 1e-15),
        extent=[0, simulation_width_x_mm, simulation_depth_z_mm, 0],
        aspect='auto',
        cmap='viridis'
    )
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("log10 Absorption map at best wavelength = " + str(best_wavelength_nm) + " nm")
    plt.colorbar(label="log10(A)")
    plt.show()

if __name__ == "__main__":
    main()