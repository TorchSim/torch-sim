"""Nudged Elastic Band (NEB) workflow.

This script demonstrates the Nudged Elastic Band method for finding minimum energy
paths between two given atomic configurations.
"""
# %%
# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "ase",
# ]
# ///

import json  # Import json for output

# Configure logging to DEBUG level first
import logging
import pickle  # Import pickle

import ase.geometry  # Import the geometry module
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk
from ase.io import read
from ase.mep import NEB as ASENEB
from ase.mep.neb import ImprovedTangentMethod, NEBState
from ase.optimize import FIRE
from mace.calculators.foundations_models import mace_mp
from mace.calculators.mace import MACECalculator
from monty.json import MontyDecoder, MontyEncoder  # Import Monty

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.state import SimState
from torch_sim.workflows.neb import NEB as TorchNEB


# Redirect logging to a file instead of stdout
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(message)s",
    filename="neb_debug.log",  # Specify the log file name
    filemode="w",
)  # Overwrite the log file each time
logging.getLogger("torch_sim.workflows.neb").setLevel(logging.DEBUG)


torch_sim_device = "cuda" if torch.cuda.is_available() else "cpu"
torch_sim_dtype = torch.float64  # Use float64 for higher precision

# Load MACE model using mace_mp like other tutorials
print("Loading MACE model...")
mace_potential = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(torch_sim_dtype).removeprefix("torch."),
    device=str(torch_sim_device),
)


def compare_initial_paths(
    ase_start_atoms,
    ase_end_atoms,
    torch_sim_initial_state: SimState,
    torch_sim_final_state: SimState,
    neb_workflow: TorchNEB,
):
    """Compares initial paths and the MIC displacement vector."""
    print("Comparing initial interpolated paths and MIC vectors...")
    n_images = neb_workflow.n_images
    n_total_images = n_images + 2
    device = neb_workflow.device
    dtype = neb_workflow.dtype

    # --- Endpoint Check ---
    print("\nChecking consistency of starting endpoint positions:")
    ase_start_pos_direct = ase_start_atoms.get_positions()
    ts_start_pos_direct = torch_sim_initial_state.positions.cpu().numpy()
    start_close = np.allclose(
        ase_start_pos_direct, ts_start_pos_direct, rtol=1e-5, atol=1e-6
    )
    print(f"  Direct Start positions close: {start_close}")
    if not start_close:
        max_diff_start = np.max(np.abs(ase_start_pos_direct - ts_start_pos_direct))
        print(f"  Max absolute difference (Start): {max_diff_start:.6f}")
    print("------------------------------------")

    # --- MIC Vector Comparison ---
    print("\nComparing Minimum Image Convention (MIC) displacement vectors:")
    # Use the torch-sim states as the source of truth for positions/cell
    raw_dr_ts = torch_sim_final_state.positions - torch_sim_initial_state.positions
    cell_ts = torch_sim_initial_state.cell[0]  # Assuming single batch cell
    pbc_ts = torch_sim_initial_state.pbc

    # ASE MIC calculation
    try:
        ase_cell_np = cell_ts.cpu().numpy()
        ase_pbc_np = np.array([pbc_ts] * 3)  # ASE expects 3 bools usually
        ase_mic_dr_np, _ = ase.geometry.find_mic(
            raw_dr_ts.cpu().numpy(), ase_cell_np, pbc=ase_pbc_np
        )
        print(f"  ASE MIC vector calculated (shape: {ase_mic_dr_np.shape})")
    except Exception as e:
        print(f"  Error calculating ASE MIC: {e}")
        ase_mic_dr_np = None

    # torch-sim MIC calculation
    try:
        ts_mic_dr = ts.transforms.minimum_image_displacement(
            dr=raw_dr_ts, cell=cell_ts, pbc=pbc_ts
        )
        ts_mic_dr_np = ts_mic_dr.cpu().numpy()
        print(f"  torch-sim MIC vector calculated (shape: {ts_mic_dr_np.shape})")
    except Exception as e:
        print(f"  Error calculating torch-sim MIC: {e}")
        ts_mic_dr_np = None

    # Compare the MIC vectors
    if ase_mic_dr_np is not None and ts_mic_dr_np is not None:
        if ase_mic_dr_np.shape != ts_mic_dr_np.shape:
            print("  Error: Shapes of MIC vectors do not match.")
        else:
            mic_vectors_close = np.allclose(
                ase_mic_dr_np, ts_mic_dr_np, rtol=1e-5, atol=1e-6
            )
            print(f"  MIC displacement vectors close: {mic_vectors_close}")
            if not mic_vectors_close:
                max_diff_mic = np.max(np.abs(ase_mic_dr_np - ts_mic_dr_np))
                norm_diff = np.linalg.norm(ase_mic_dr_np - ts_mic_dr_np)
                print(f"  Max absolute difference (MIC vectors): {max_diff_mic:.6f}")
                print(f"  Norm of difference vector (MIC): {norm_diff:.6f}")
                print("  This difference likely causes the interpolation discrepancy.")
    print("------------------------------------")

    # --- Get ASE interpolated path ---
    ase_images = [ase_start_atoms.copy() for _ in range(n_images + 1)]
    ase_images.append(ase_end_atoms.copy())
    ase_neb_calc = ASENEB(ase_images, climb=False)
    ase_neb_calc.interpolate(mic=True)
    ase_positions = np.stack([img.get_positions() for img in ase_neb_calc.images])
    print(f"\n  ASE interpolated path shape: {ase_positions.shape}")

    # --- Get torch-sim interpolated path ---
    try:
        interpolated_state = neb_workflow._interpolate_path(
            torch_sim_initial_state, torch_sim_final_state
        )
        ts_interp_pos = interpolated_state.positions
        ts_start_pos = torch_sim_initial_state.positions
        ts_end_pos = torch_sim_final_state.positions
        n_atoms = ts_start_pos.shape[0]
        ts_interp_pos_reshaped = ts_interp_pos.reshape(n_images, n_atoms, 3)
        ts_positions = torch.cat(
            [
                torch_sim_initial_state.positions.unsqueeze(0).to(device, dtype),
                ts_interp_pos_reshaped.to(device, dtype),
                torch_sim_final_state.positions.unsqueeze(0).to(device, dtype),
            ],
            dim=0,
        )
        ts_positions_np = ts_positions.cpu().numpy()
        print(f"  torch-sim interpolated path shape (direct): {ts_positions_np.shape}")
    except Exception as e:
        print(f"  Error during torch-sim interpolation: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- Compare Interpolated Paths ---
    print(
        "\n  Per-image comparison of interpolated paths (Max Abs Error | Mean Abs Error):"
    )
    overall_max_diff_interp = 0.0
    if ase_positions.shape != ts_positions_np.shape:
        print("  Error: Shapes of ASE and torch-sim interpolated paths do not match.")
        return

    for i in range(n_total_images):
        diff_image_i = np.abs(ase_positions[i] - ts_positions_np[i])
        max_ae_i = np.max(diff_image_i)
        mae_i = np.mean(diff_image_i)
        print(f"    Image {i}: MaxAE = {max_ae_i:.6f} | MAE = {mae_i:.6f}")
        overall_max_diff_interp = max(overall_max_diff_interp, max_ae_i)

    are_close_interp = np.allclose(ase_positions, ts_positions_np, rtol=1e-5, atol=1e-6)

    if are_close_interp:
        print("  Overall: Interpolated paths are numerically close.")
    else:
        print("  Overall: Interpolated paths differ numerically.")
        print(
            f"  Overall Maximum absolute difference (Interpolated): {overall_max_diff_interp:.6f}"
        )


def ase_neb(start_atoms, end_atoms, nimages=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = [start_atoms.copy() for _ in range(nimages + 1)]
    images.append(end_atoms.copy())

    neb_calc = ASENEB(images, climb=True, method="improvedtangent")
    neb_calc.interpolate(mic=True)

    # Attach calculator to all images using mace_mp
    ase_dtype_str = "float64" if torch_sim_dtype == torch.float64 else "float32"
    print(f"Attaching ASE calculator with dtype: {ase_dtype_str} to all images")
    ase_calc = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        device=device,
        default_dtype=ase_dtype_str,
        dispersion=False,
    )
    for image in neb_calc.images:
        image.calc = ase_calc

    # Set up trajectory logging for the reference ASE run (Commented out as not used for plot)
    # ase_traj_filename = "ase_ref_neb.traj"
    opt = FIRE(neb_calc)
    # opt.attach(traj)       # Attach the trajectory logger

    # Run the ASE optimization (essential)
    print("Running ASE NEB optimization...")
    opt.run(fmax=0.05, steps=1000)
    print("Finished ASE NEB optimization.")

    return neb_calc  # Only return the final NEB object


def relax_atoms(
    atoms,
    fmax=0.05,
    steps=1000,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
):
    new_atoms = atoms.copy()
    ase_dtype_str = "float64" if dtype == torch.float64 else "float32"
    new_atoms.calc = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        device=str(device),
        default_dtype=ase_dtype_str,
        dispersion=False,
    )
    opt = FIRE(new_atoms)
    opt.run(fmax=fmax, steps=steps)
    return new_atoms


# Create the torch_sim wrapper
ts_mace_model = MaceModel(
    model=mace_potential,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    compute_forces=True,  # Default, but good to be explicit
    compute_stress=True,  # Needed by interface if we want stress later
    enable_cueq=False,
)

# initial_trajectory = read('/home/myless/Packages/forge/scratch/data/neb_workflow_data/Cr7Ti8V104W8Zr_Cr_to_V_site102_to_69_initial.xyz', index=':')
# print(len(initial_trajectory))

# Create simple test structures for demonstration
# Using bulk structures instead of file paths
# Create simple test structures (can be replaced with file reads if needed)
start_atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
end_atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
# Add a small displacement to create a path
end_atoms.positions[0] += [0.1, 0.1, 0.1]

relaxed_start_atoms = relax_atoms(start_atoms)
relaxed_end_atoms = relax_atoms(end_atoms)

traj_file_name = "neb_path_torchsim_fire_5im.hdf5"

# --- Setup ASE NEB for comparison ---
n_intermediate_images_ase = 5
ase_images_compare = [relaxed_start_atoms.copy()]
ase_images_compare.extend(
    [relaxed_start_atoms.copy() for _ in range(n_intermediate_images_ase)]
)
ase_images_compare.append(relaxed_end_atoms.copy())

ase_neb_compare = ASENEB(
    ase_images_compare,
    k=0.1,  # Match torch-sim spring constant
    climb=True,  # Match torch-sim setting
    method="improvedtangent",  # Match torch-sim tangent method
)
ase_neb_compare.interpolate(mic=True)  # Initial interpolation

device = "cuda" if torch.cuda.is_available() else "cpu"
# Attach calculator to ALL ASE images using mace_mp
ase_dtype_str_compare = "float64" if torch_sim_dtype == torch.float64 else "float32"
print(f"Using ASE comparison calculator dtype: {ase_dtype_str_compare}")
ase_calculator = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    device=device,
    default_dtype=ase_dtype_str_compare,
    dispersion=False,
)
for img in ase_neb_compare.images:
    img.calc = ase_calculator
# ----------------------------------

initial_system = ts.io.atoms_to_state(
    relaxed_start_atoms.copy(), device=torch_sim_device, dtype=torch_sim_dtype
)
final_system = ts.io.atoms_to_state(
    relaxed_end_atoms.copy(), device=torch_sim_device, dtype=torch_sim_dtype
)

neb_workflow = TorchNEB(
    model=ts_mace_model,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    spring_constant=0.1,
    n_images=5,
    use_climbing_image=True,  # Set as desired for the actual run
    optimizer_type="ase_fire",  # Set as desired for the actual run
    optimizer_params={},
    trajectory_filename=traj_file_name,
)

compare_initial_paths(
    relaxed_start_atoms, relaxed_end_atoms, initial_system, final_system, neb_workflow
)


# --- Add Function for Manual ASE Force Calculation ---
def calculate_ase_neb_force_step0(
    ase_neb_calc: ASENEB,
    image_index: int,
    neb_workflow: TorchNEB,
    output_filename="ase_step0_debug.json",
):
    """Manually calculates the ASE NEB force components for a specific
    intermediate image at step 0 (after initial interpolation) and saves
    the results to a JSON file.
    Uses the ImprovedTangent method for consistency with torch-sim default.
    """
    print(f"--- Calculating ASE NEB Debug Info (Step 0, Image Index {image_index}) ---")
    debug_data = {
        "step": 0,
        "image_index_intermediate": image_index - 1,  # 0-based index among intermediates
        "image_index_absolute": image_index,  # 0-based index in full list
        "inputs": {},
        "outputs": {},
        "error": None,
    }

    n_images = ase_neb_calc.nimages  # Total number of images including endpoints
    if not (0 < image_index < n_images - 1):
        error_msg = f"Error: image_index {image_index} is not an intermediate image."
        print(error_msg)
        debug_data["error"] = error_msg
        with open(output_filename, "w") as f:
            json.dump(debug_data, f, indent=2, cls=MontyEncoder)  # Use MontyEncoder
        return

    # 1. Get initial energies and forces after interpolation + calculator attachment
    try:
        initial_energies_np = np.array(
            [img.get_potential_energy() for img in ase_neb_calc.images]
        )
        initial_forces_np = np.stack([img.get_forces() for img in ase_neb_calc.images])

        # No need for .tolist() with MontyEncoder
        debug_data["inputs"]["energies_all"] = initial_energies_np
        debug_data["inputs"]["true_forces_image"] = initial_forces_np[image_index]
        debug_data["inputs"]["positions_image_minus_1"] = ase_neb_calc.images[
            image_index - 1
        ].get_positions()
        debug_data["inputs"]["positions_image"] = ase_neb_calc.images[
            image_index
        ].get_positions()
        debug_data["inputs"]["positions_image_plus_1"] = ase_neb_calc.images[
            image_index + 1
        ].get_positions()
        debug_data["inputs"]["cell"] = (
            ase_neb_calc.images[image_index].get_cell().tolist()
        )
        # No need for bool() conversion with MontyEncoder
        debug_data["inputs"]["pbc"] = ase_neb_calc.images[image_index].get_pbc()

    except Exception as e:
        error_msg = f"Error getting initial energies/forces from ASE images: {e}"
        print(error_msg)
        debug_data["error"] = error_msg
        import traceback

        debug_data["traceback"] = traceback.format_exc()
        with open(output_filename, "w") as f:
            json.dump(debug_data, f, indent=2, cls=MontyEncoder)  # Use MontyEncoder
        return

    # 2. Setup NEB state and method objects
    ase_neb_obj_for_state = ASENEB(
        ase_neb_calc.images,
        k=neb_workflow.spring_constant,
        climb=neb_workflow.use_climbing_image,
        method="improvedtangent",
    )
    neb_state = NEBState(ase_neb_obj_for_state, ase_neb_calc.images, initial_energies_np)
    tangent_method = ImprovedTangentMethod(ase_neb_obj_for_state)

    # 3. Calculate components for the target image_index
    try:
        spring1 = neb_state.spring(image_index - 1)
        spring2 = neb_state.spring(image_index)
        # No .tolist() needed
        debug_data["outputs"]["mic_displacement_1"] = spring1.t
        debug_data["outputs"]["mic_displacement_2"] = spring2.t

        # Calculate tangent
        tangent_ase = tangent_method.get_tangent(neb_state, spring1, spring2, image_index)
        tangent_norm_ase = np.linalg.norm(tangent_ase)
        if tangent_norm_ase > 1e-15:
            tangent_ase_normalized = tangent_ase / tangent_norm_ase
        else:
            tangent_ase_normalized = tangent_ase  # Keep as zero vector
        tangent_norm_final = np.linalg.norm(tangent_ase_normalized)

        # No .tolist() needed
        debug_data["outputs"]["tangent_vector"] = tangent_ase_normalized
        debug_data["outputs"]["tangent_norm"] = tangent_norm_final

        # Calculate perpendicular force
        true_force_img = initial_forces_np[image_index]
        f_true_dot_tau_ase = np.vdot(true_force_img, tangent_ase_normalized)
        f_perp_ase = true_force_img - f_true_dot_tau_ase * tangent_ase_normalized
        f_perp_norm = np.linalg.norm(f_perp_ase)

        # No .tolist() needed
        debug_data["outputs"]["f_true_dot_tau"] = f_true_dot_tau_ase
        debug_data["outputs"]["f_perp_vector"] = f_perp_ase
        debug_data["outputs"]["f_perp_norm"] = f_perp_norm

        # Calculate parallel spring force
        segment_lengths_all = [neb_state.spring(i).nt for i in range(n_images - 1)]
        spring_mag_term = spring2.nt * spring2.k - spring1.nt * spring1.k
        f_spring_par_ase = spring_mag_term * tangent_ase_normalized
        f_spring_par_norm = np.linalg.norm(f_spring_par_ase)

        # No .tolist() needed
        debug_data["outputs"]["segment_lengths"] = segment_lengths_all
        debug_data["outputs"]["spring_force_magnitude_term"] = spring_mag_term
        debug_data["outputs"]["f_spring_par_vector"] = f_spring_par_ase
        debug_data["outputs"]["f_spring_par_norm"] = f_spring_par_norm

        # Calculate total NEB force (before potential climbing modification)
        neb_force_ase = f_perp_ase + f_spring_par_ase
        # Explicitly convert to numpy array before saving, remove .tolist()
        debug_data["outputs"]["neb_force_before_climb_vector"] = np.array(neb_force_ase)
        debug_data["outputs"]["neb_force_before_climb_norm"] = np.linalg.norm(
            neb_force_ase
        )

        # --- Direct Debug Prints for Step 0 ---
        print("\n  --- DIRECT DEBUG PRINT (ASE STEP 0) ---")
        print(f"    f_perp_norm: {f_perp_norm}")
        print(f"    f_perp_vec[0]: {f_perp_ase[0]}")
        print(f"    spring1_length (R[{image_index}]-R[{image_index - 1}]): {spring1.nt}")
        print(f"    spring2_length (R[{image_index + 1}]-R[{image_index}]): {spring2.nt}")
        print(f"    Length Diff (spring2.nt - spring1.nt): {spring2.nt - spring1.nt}")
        print(f"    f_spring_par_norm: {f_spring_par_norm}")
        print(f"    f_spring_par_vec[0]: {f_spring_par_ase[0]}")
        print(f"    neb_force_before_climb_norm: {np.linalg.norm(neb_force_ase)}")
        print("  ------------------------------------")
        # --------------------------------------

        # Handle climbing image modification
        is_climbing = ase_neb_obj_for_state.climb and image_index == neb_state.imax
        debug_data["outputs"]["is_climbing_image"] = is_climbing
        debug_data["outputs"]["imax"] = int(
            neb_state.imax
        )  # Ensure imax is JSON serializable

        if is_climbing:
            climbing_force_ase = (
                true_force_img - 2 * f_true_dot_tau_ase * tangent_ase_normalized
            )
            climbing_force_norm = np.linalg.norm(climbing_force_ase)
            # No .tolist() needed
            debug_data["outputs"]["climbing_force_vector"] = climbing_force_ase
            debug_data["outputs"]["climbing_force_norm"] = climbing_force_norm
            final_force_ase = climbing_force_ase
        else:
            final_force_ase = neb_force_ase

        # No .tolist() needed
        debug_data["outputs"]["final_neb_force_vector"] = final_force_ase
        debug_data["outputs"]["final_neb_force_norm"] = np.linalg.norm(final_force_ase)

    except Exception as e:
        error_msg = (
            f"Error during manual ASE force calculation for image {image_index}: {e}"
        )
        print(error_msg)
        debug_data["error"] = error_msg
        import traceback

        debug_data["traceback"] = traceback.format_exc()

    # Write data to JSON
    try:
        with open(output_filename, "w") as f:
            json.dump(debug_data, f, indent=2, cls=MontyEncoder)  # Use MontyEncoder
        print(f"--- ASE NEB Debug Info saved to {output_filename} ---")
    except Exception as e:
        print(f"Error writing ASE debug info to JSON: {e}")


# --- Add Function for Comparing JSON/Pickle Outputs to debug the tangent force calculation ---
def compare_step0_outputs(
    file_ase="ase_step0_debug.json",
    file_ts="torchsim_step0_debug.pkl",
    rtol=1e-5,
    atol=1e-6,
):
    print("\n--- Comparing Step 0 Debug Outputs (ASE JSON vs TorchSim Pickle) --- ")
    try:
        # Load ASE data from JSON
        with open(file_ase) as f:
            data_ase = json.load(f, cls=MontyDecoder)
        # Load TorchSim data from Pickle
        with open(file_ts, "rb") as f:  # Use 'rb' for pickle
            data_ts = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return
    except Exception as e:
        print(f"Error loading JSON/Pickle files: {e}")
        return

    # Basic checks
    if data_ase.get("error") or data_ts.get("error"):
        print("Comparison aborted due to error during data generation.")
        print(f"  ASE Error: {data_ase.get('error')}")
        print(f"  TS Error: {data_ts.get('error')}")
        return

    if data_ase.get("step") != 0 or data_ts.get("step") != 0:
        print("Warning: One or both files do not contain step 0 data.")
        # Continue comparison anyway

    if data_ase.get("image_index_intermediate") != data_ts.get(
        "image_index_intermediate"
    ):
        print("Warning: JSON files are for different intermediate image indices.")
        # Continue comparison anyway

    outputs_ase = data_ase.get("outputs", {})
    outputs_ts = data_ts.get("outputs", {})

    all_keys = set(outputs_ase.keys()) | set(outputs_ts.keys())
    mismatches = 0

    print(
        f"Comparing fields for intermediate image index: {data_ase.get('image_index_intermediate', 'N/A')}"
    )

    for key in sorted(list(all_keys)):
        val_ase = outputs_ase.get(key)
        val_ts = outputs_ts.get(key)

        if key not in outputs_ts:
            print(f"  - Key '{key}': Present in ASE, Missing in TorchSim")
            mismatches += 1
            continue
        if key not in outputs_ase:
            print(f"  - Key '{key}': Missing in ASE, Present in TorchSim")
            mismatches += 1
            continue

        # --- Handle Type Conversion for Comparison ---
        val_ase_comp = val_ase
        val_ts_comp = val_ts

        # Convert torch tensor from pickle to numpy/scalar for comparison
        if isinstance(val_ts_comp, torch.Tensor):
            if val_ts_comp.ndim == 0:  # Scalar tensor
                val_ts_comp = val_ts_comp.item()
            else:
                val_ts_comp = val_ts_comp.detach().cpu().numpy()  # Use detach()
        # --------------------------------------------

        # --- Debug Print for Specific Key ---
        if key == "neb_force_before_climb_vector":
            print(
                f"    DEBUG compare [{key}]: ASE[0]={np.array(val_ase_comp)[0]}, TS[0]={np.array(val_ts_comp)[0]}"
            )
        # ------------------------------------

        # --- Special Handling for imax index ---
        if key == "imax":
            # ASE imax is index in full list (1 to n_images-1)
            # TS imax is index in intermediates (0 to n_images-2)
            # Compare ASE imax with TS imax + 1
            ase_imax = int(val_ase_comp)
            ts_imax_plus_1 = int(val_ts_comp) + 1
            match = ase_imax == ts_imax_plus_1
            if not match:
                difference_info = f"ASE imax={ase_imax}, TS imax(adj)={ts_imax_plus_1}"
            status = "Match" if match else "DIFFER"
            print(f"  - Key '{key:<30}': {status}  {difference_info}")
            if not match:
                mismatches += 1
            continue  # Skip rest of comparison for imax
        # -------------------------------------

        # Try numerical comparison first
        match = False
        difference_info = ""
        try:
            # Ensure they are numpy arrays for consistent comparison
            # ASE data might already be numpy or list, TS data was converted above
            arr_ase = np.array(val_ase_comp)
            arr_ts = np.array(val_ts_comp)

            if arr_ase.shape != arr_ts.shape:
                match = False
                difference_info = f"Shapes differ: ASE={arr_ase.shape}, TS={arr_ts.shape}"
            elif np.issubdtype(arr_ase.dtype, np.number) and np.issubdtype(
                arr_ts.dtype, np.number
            ):
                match = np.allclose(arr_ase, arr_ts, rtol=rtol, atol=atol)
                if not match:
                    max_abs_diff = np.max(np.abs(arr_ase - arr_ts))
                    difference_info = f"Max abs diff: {max_abs_diff:.6e}"
            elif arr_ase.dtype == np.bool_ and arr_ts.dtype == np.bool_:
                match = np.array_equal(arr_ase, arr_ts)
                if not match:
                    difference_info = f"Boolean values differ: ASE={arr_ase}, TS={arr_ts}"
            else:  # Fallback for other types (e.g., strings if they were arrays)
                match = np.array_equal(arr_ase, arr_ts)
                if not match:
                    difference_info = "Non-numerical array values differ"

        except (TypeError, ValueError):
            # Fallback to direct comparison for non-array types or incompatible arrays
            try:
                if isinstance(val_ase_comp, (float, int)) and isinstance(
                    val_ts_comp, (float, int)
                ):
                    match = np.isclose(val_ase_comp, val_ts_comp, rtol=rtol, atol=atol)
                    if not match:
                        difference_info = f"Diff: {abs(val_ase_comp - val_ts_comp):.6e}"
                elif type(val_ase_comp) == type(val_ts_comp):
                    match = val_ase_comp == val_ts_comp
                    if not match:
                        difference_info = (
                            f"Values differ: ASE='{val_ase_comp}', TS='{val_ts_comp}'"
                        )
                else:
                    # Types should ideally match after conversion, but check just in case
                    match = False
                    difference_info = f"Types differ after conversion: ASE={type(val_ase_comp)}, TS={type(val_ts_comp)}"
            except Exception:
                match = False

        status = "Match" if match else "DIFFER"  # Pad DIFFER for alignment
        print(f"  - Key '{key:<30}': {status}  {difference_info}")
        if not match:
            mismatches += 1

    if mismatches == 0:
        print("\nAll compared output fields match.")
    else:
        print(f"\nFound {mismatches} mismatch(es) in output fields.")
    print("--- End Comparison --- ")


# -------------------------------------------------


# --- Add Function to Print Pickle Structure ---
def print_pickle_structure(filename="torchsim_step0_debug.pkl"):
    print(f"\n--- Structure of Pickle File: {filename} --- ")
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    if not isinstance(data, dict):
        print(f"Loaded data is not a dictionary (Type: {type(data)})")
        return

    print(f"Keys: {list(data.keys())}")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                val_type = type(subvalue)
                val_shape = getattr(subvalue, "shape", "N/A")
                # Add dtype for tensors
                val_dtype = getattr(subvalue, "dtype", "N/A")
                print(
                    f"    - {subkey:<30}: Type={val_type}, Shape={val_shape}, Dtype={val_dtype}"
                )
        else:
            val_type = type(value)
            val_shape = getattr(value, "shape", "N/A")
            val_dtype = getattr(value, "dtype", "N/A")
            print(f"  {key:<32}: Type={val_type}, Shape={val_shape}, Dtype={val_dtype}")
    print("--- End Pickle Structure --- ")


# --------------------------------------------

# --- Perform manual ASE force calculation for step 0 ---
debug_ase_img_index = (
    n_intermediate_images_ase // 2 + 1
)  # Index in the full list (0 to n_images+1)
calculate_ase_neb_force_step0(ase_neb_compare, debug_ase_img_index, neb_workflow)
# ------------------------------------------------------

print("\nStarting torch-sim NEB optimization...")
final_path_gd = neb_workflow.run(
    initial_system=initial_system,
    final_system=final_system,
    max_steps=100,  # Keep increased steps for now
    fmax=0.05,
)
print("Finished torch-sim NEB optimization.")

# Check if it converged and plot results
results = ts_mace_model(
    dict(
        positions=final_path_gd.positions,
        cell=final_path_gd.cell,
        atomic_numbers=final_path_gd.atomic_numbers,
        system_idx=final_path_gd.system_idx,
        pbc=True,
    )
)

energies = results["energy"].tolist()

# Including the energies from the ASE NEB calculation for comparison
# ase_energies = [0.0, 0.154541015625, 0.6151123046875, 0.8592529296875, 0.8148193359375, 0.5965576171875, 0.47705078125]

ase_neb_calc = ase_neb(relaxed_start_atoms, relaxed_end_atoms, nimages=5)
ase_energies = [image.get_potential_energy() for image in ase_neb_calc.images]
scaled_ase_energies = [e - ase_energies[0] for e in ase_energies]


scaled_energies = [e - energies[0] for e in energies]

print(scaled_energies)
torch_sim_barrier = max(scaled_energies) - scaled_energies[0]
ase_barrier = max(scaled_ase_energies) - scaled_ase_energies[0]

# Create normalized reaction coordinates (0 to 1) for both datasets
torch_sim_coords = np.linspace(0, 1, len(scaled_energies))
ase_coords = np.linspace(0, 1, len(scaled_ase_energies))

# Create a common x-axis with 100 points for smoother plotting
common_coords = np.linspace(0, 1, 100)

# Interpolate both energy profiles to the common coordinate system
torch_sim_interp = np.interp(common_coords, torch_sim_coords, scaled_energies)
ase_interp = np.interp(common_coords, ase_coords, scaled_ase_energies)

# --- Print Pickle Structure to Verify ---
# print_pickle_structure()
# -------------------------------------

# --- Compare Step 0 Debug Outputs for compute_tangent at step 0---
# compare_step0_outputs() # Use the updated function name
# ------------------------------------


# --- Plot the energy profiles ---
plt.plot(common_coords, torch_sim_interp, label="torch-sim")
plt.plot(common_coords, ase_interp, label="ASE")
plt.xlabel("Reaction Coordinate")
plt.ylabel("Energy (eV)")
plt.title(
    f"ASE Barrier = {ase_barrier:.4f} eV, torch-sim Barrier = {torch_sim_barrier:.4f} eV, Difference = {torch_sim_barrier - ase_barrier:.4f} eV"
)
plt.legend()
plt.show()
# ------------------------------------


# --- Function to Inspect HDF5 File Structure ---
def inspect_hdf5(filename):
    print(f"\n--- Inspecting HDF5 File: {filename} ---")
    try:
        with h5py.File(filename, "r") as f:

            def print_attrs(name, obj):
                print(f"  Path: /{name}")
                if isinstance(obj, h5py.Dataset):
                    print("    Type: Dataset")
                    print(f"    Shape: {obj.shape}")
                    print(f"    Dtype: {obj.dtype}")
                    # Optionally print a small slice of data
                    # try:
                    #     print(f"    Data sample: {obj[0:min(2, obj.shape[0])]}")
                    # except Exception as e:
                    #     print(f"    Could not read data sample: {e}")
                elif isinstance(obj, h5py.Group):
                    print("    Type: Group")
                print(f"    Attributes: {dict(obj.attrs)}")

            f.visititems(print_attrs)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")
    print("--- End HDF5 Inspection ---")


# ----------------------------------------------


# --- Analyze Optimizer Convergence ---
def analyze_convergence(ts_traj_file, ase_fmax_csv_file):
    print("\n--- Analyzing Optimizer Convergence ---")
    max_force_ts = []
    max_force_ase = []

    # Analyze torch-sim trajectory
    try:
        with h5py.File(ts_traj_file, "r") as f:
            if "data/neb_forces" not in f or "data/image_indices" not in f:
                raise ValueError(
                    "HDF5 file missing '/data/neb_forces' or '/data/image_indices' datasets."
                )

            # Data is under /data group, steps are the first dimension
            neb_forces_dset = f["/data/neb_forces"]
            image_indices_dset = f["/data/image_indices"]

            n_steps = neb_forces_dset.shape[0]
            # Read static image indices (take the first slice)
            image_indices = image_indices_dset[0, :]

            # Infer dimensions
            n_images_total = len(np.unique(image_indices))
            n_atoms_total = len(image_indices)
            if neb_forces_dset.shape[1] != n_atoms_total:
                raise ValueError(
                    f"Mismatch between image_indices length ({n_atoms_total}) and neb_forces second dimension ({neb_forces_dset.shape[1]})"
                )

            n_atoms_per_image = n_atoms_total // n_images_total
            print(
                f"TorchSim Traj: {n_steps} steps, {n_images_total} total images, {n_atoms_per_image} atoms/image."
            )

            for step in range(n_steps):
                # Access forces for the current step from the first dimension
                neb_forces = torch.from_numpy(neb_forces_dset[step, :, :])

                # Select forces only for intermediate images (index 1 to n_images_total - 2)
                intermediate_mask = (image_indices > 0) & (
                    image_indices < n_images_total - 1
                )
                forces_intermediate = neb_forces[intermediate_mask]
                if forces_intermediate.numel() > 0:
                    max_comp = torch.max(torch.abs(forces_intermediate)).item()
                    max_force_ts.append(max_comp)
                else:
                    max_force_ts.append(0.0)  # Or handle error/empty case

    except Exception as e:
        print(f"Error reading torch-sim trajectory {ts_traj_file}: {e}")

    # Read ASE fmax data from CSV
    try:
        # Use numpy.loadtxt to read the 2nd column (index 1) from the CSV
        # Assuming tab delimiter, skipping header row
        ase_data = np.loadtxt(ase_fmax_csv_file, delimiter="\t", skiprows=1, usecols=(1,))
        max_force_ase = ase_data.tolist()  # Convert numpy array to list
        print(f"Read {len(max_force_ase)} fmax values from {ase_fmax_csv_file}")
    except Exception as e:
        print(f"Error reading ASE fmax CSV file {ase_fmax_csv_file}: {e}")

    # Plotting
    if max_force_ts or max_force_ase:
        plt.figure()
        if max_force_ts:
            plt.plot(
                range(len(max_force_ts)),
                max_force_ts,
                label="torch-sim (ase_fire)",
                marker=".",
            )
        if max_force_ase:
            plt.plot(
                range(len(max_force_ase)), max_force_ase, label="ASE (FIRE)", marker="."
            )
        plt.xlabel("Optimization Step")
        plt.ylabel("Max Abs Force Component (eV/Ang)")
        plt.title("Optimizer Convergence Comparison")
        plt.legend()
        plt.grid(True)
        plt.yscale("log")  # Log scale often helpful for forces
        plt.show()
    else:
        print("No force data extracted to plot convergence.")


# inspect_hdf5(traj_file_name)
analyze_convergence(traj_file_name, "ase_fmax_convergence.csv")
# ---------------------------------

# --- Debugging Functions (Keep for reference) ---
# def calculate_ase_neb_force_step0(...): ...
# def compare_step0_outputs(...): ...
# def print_pickle_structure(...): ...


# --- Call Step 0 Debug Functions (Commented out) ---
# # Perform manual ASE force calculation for step 0
debug_ase_img_index = n_intermediate_images_ase // 2 + 1
calculate_ase_neb_force_step0(ase_neb_compare, debug_ase_img_index, neb_workflow)

# # Print Pickle Structure to Verify
print_pickle_structure()

# # Compare Step 0 Debug Outputs
compare_step0_outputs()
# --------------------------------------------------
