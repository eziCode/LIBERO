# Graph Report - scripts  (2026-08-12)

## Corpus Check
- 61 files · ~360,217 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 401 nodes · 594 edges · 55 communities (44 shown, 11 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- ManiSkill MuJoCo Retargeting
- Demonstration Collection
- Wavelet Band Analysis
- Force Torque Extraction
- ManiSkill Robomimic Conversion
- UniVTAC MuJoCo Conversion
- Xbox Controller Interface
- UniVTAC LIBERO Packaging
- Demonstration Data Model
- ManiSkill LIBERO Packaging
- Dataset Creation Tools
- HDF5 Attribute Enrichment
- LIBERO Task Example
- ManiSkill Trajectory Rendering
- Wavelet Burst Analysis
- LIBERO Data Processing
- UniVTAC Trajectory Rendering
- Isolated Shake Analysis
- Action Fidelity Analysis
- Motion Quality Analysis
- Task Template Generation
- MimicGen Force Videos
- Environment Contact Sheets
- Harmonic Sweep Analysis
- Wavelet Hammering Analysis
- USD Mesh Extraction
- HDF5 Inspection
- Action Distribution Analysis
- UniVTAC Validation
- Trajectory Dashboard
- Spectral Plot Generation
- Xbox Axis Testing
- Harmonic Dashboard
- Isolated Shake Dashboard
- ForceVLA Dataset Download
- TAF Dataset Download
- Video Overlay Tools
- Wavelet Hammering Dashboard
- Dataset Integrity Checks
- LIBERO Format Repair
- Shake Cup Format Repair
- Dataset Information
- Xbox Button Testing
- Xbox Raw Diagnostics

## God Nodes (most connected - your core abstractions)
1. `main()` - 15 edges
2. `convert()` - 13 edges
3. `XboxControllerBluetooth` - 13 edges
4. `XboxControllerHID` - 12 edges
5. `map_trajectory_states()` - 11 edges
6. `convert_episode()` - 10 edges
7. `replay_trajectory()` - 10 edges
8. `convert()` - 9 edges
9. `migrate_demo()` - 8 edges
10. `extract_file()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Demonstration Trajectory Schema` --conceptually_related_to--> `LIBERO Demonstration State Dimensions`  [INFERRED]
  inspect_output.txt → states_dims.txt
- `Proprioceptive Observations` --conceptually_related_to--> `Task-Dependent State Representation`  [INFERRED]
  inspect_output.txt → states_dims.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LIBERO Task State Variability** — scripts_states_dims_kitchen_scene_tasks, scripts_states_dims_living_room_scene_tasks, scripts_states_dims_study_scene_tasks, scripts_states_dims_unique_state_dimensions [EXTRACTED 1.00]
- **Shake Cup Demonstration Stack** — scripts_inspect_output_libero_tabletop_manipulation, scripts_inspect_output_panda_robot, scripts_inspect_output_osc_pose_controller, scripts_inspect_output_mujoco_simulation_model, scripts_inspect_output_demonstration_trajectory_schema [EXTRACTED 1.00]

## Communities (55 total, 11 thin omitted)

### Community 0 - "ManiSkill MuJoCo Retargeting"
Cohesion: 0.14
Nodes (34): convert(), copy_initial_object_poses(), finger_contact_forces(), libero_actions(), main(), make_environment(), map_trajectory_states(), natural_trajectory_key() (+26 more)

### Community 1 - "Demonstration Collection"
Cohesion: 0.09
Nodes (14): collect_human_trajectory(), gather_demonstrations_as_hdf5(), Gathers the demonstrations saved in @directory into a single hdf5 file. The…, Collect a demonstration with debug logging., Device, Background thread that continuously reads HID reports., Parse Bluetooth HID report and update current control values., Read unsigned 16-bit LE, convert to -1.0..1.0 centered at 32768. (+6 more)

### Community 2 - "Wavelet Band Analysis"
Cohesion: 0.25
Nodes (20): bootstrap_band_contrasts(), bootstrap_event_series(), bootstrap_post_window(), derive_pressure_regimes(), event_onsets(), main(), parse_args(), plot_average_scalograms() (+12 more)

### Community 3 - "Force Torque Extraction"
Cohesion: 0.18
Nodes (18): extract_file(), gripper_contact_forces(), initialize_output(), localize_robosuite_assets(), main(), make_environment(), natural_demo_key(), parse_args() (+10 more)

### Community 4 - "ManiSkill Robomimic Conversion"
Cohesion: 0.22
Nodes (18): convert(), datasets(), episode_metadata(), flattened_state(), load_metadata(), natural_key(), observation_components(), parse_args() (+10 more)

### Community 5 - "UniVTAC MuJoCo Conversion"
Cohesion: 0.26
Nodes (16): convert_episode(), decode_rgb(), main(), make_env(), merge_obs(), model_xml_with_slot_pose(), normalized_joint_target(), numeric_obs() (+8 more)

### Community 6 - "Xbox Controller Interface"
Cohesion: 0.14
Nodes (9): Device, Parse GIP report and update current control values., Xbox controller wrapper using hidapi (GIP protocol, report ID 0x20). Uses the…, Read a signed 16-bit LE value and normalize to -1.0..1.0., Returns current controller state (SpaceMouse pattern). Joystick position =…, Apply deadzone with scaled output., Open the HID device and start the background reader thread., Background thread that continuously reads HID reports. (+1 more)

### Community 7 - "UniVTAC LIBERO Packaging"
Cohesion: 0.25
Nodes (14): libero_actions(), main(), migrate_demo(), natural(), Dataset, Group, ndarray, Embed this demo's fixed slot pose and active recorded grasp in its XML. (+6 more)

### Community 8 - "Demonstration Data Model"
Cohesion: 0.13
Nodes (16): Demonstration Trajectory Schema, IsGrasped Checkpoint, LIBERO Tabletop Manipulation, MuJoCo Simulation Model, OSC Pose Controller, Panda Robot, Proprioceptive Observations, Shake Cup Demonstration Dataset (+8 more)

### Community 9 - "ManiSkill LIBERO Packaging"
Cohesion: 0.32
Nodes (13): add_images(), canonical_actions(), finger_forces(), hard_link(), main(), make_env(), migrate(), migrate_demo() (+5 more)

### Community 10 - "Dataset Creation Tools"
Cohesion: 0.17
Nodes (7): get_left_right_gripper_contact_force(), main(), collect_human_trajectory(), gather_demonstrations_as_hdf5(), Modified from robosuite example scripts. A script to collect a batch of human…, Gathers the demonstrations saved in @directory into a single hdf5 file. The…, Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration. The…

### Community 11 - "HDF5 Attribute Enrichment"
Cohesion: 0.31
Nodes (10): AttributeManager, attributes_equal(), enrich_file(), main(), original_structure(), parse_args(), File, Namespace (+2 more)

### Community 12 - "LIBERO Task Example"
Cohesion: 0.22
Nodes (4): KitchenScene1, This is a standalone file for create a task in libero., InitialSceneTemplates, register_mu

### Community 13 - "ManiSkill Trajectory Rendering"
Cohesion: 0.33
Nodes (8): finger_contact_forces(), force_overlay(), main(), ndarray, Path, Return summed normal contact force for each Panda finger in Newtons., Draw readable wrench text and tactile bar gauges on an RGB frame., render_dataset()

### Community 14 - "Wavelet Burst Analysis"
Cohesion: 0.43
Nodes (7): analyze_bursts(), calculate_wavelet_energies(), create_synced_video(), extract_physical_events(), Decomposes actions into low, mid, and high frequency components., run_experiment(), upsample_signal()

### Community 15 - "LIBERO Data Processing"
Cohesion: 0.36
Nodes (7): main, dump_demo(), main(), process_demo(), process_task_dataset(), Generate preprocessed data for a task., Collect gt depths in simulation by replaying demos

### Community 16 - "UniVTAC Trajectory Rendering"
Cohesion: 0.39
Nodes (7): contact_forces(), main(), ndarray, Path, Compose the MuJoCo view with original left/right GelSight observations., render(), tactile_overlay()

### Community 17 - "Isolated Shake Analysis"
Cohesion: 0.43
Nodes (6): check_checkpoint(), process_hybrid_sequence(), Applies absolute band filtering to an action chunk., Keeps everything UNFILTERED until the grasp_idx. After the grasp, applies the…, run_isolated_experiment(), spectral_filter_bands()

### Community 18 - "Action Fidelity Analysis"
Cohesion: 0.47
Nodes (5): process_full_sequence(), Applies DCT, zeros out high frequency coefficients, and applies IDCT.…, Processes the entire action sequence in chunks. chunk_size: int, 'Full', or…, run_experiment(), spectral_filter()

### Community 19 - "Motion Quality Analysis"
Cohesion: 0.53
Nodes (5): analyze_hammering_impact(), analyze_mixing_diagonal(), analyze_rotation_continuity(), Applies a heavy spectral filter to the entire continuous sequence., spectral_filter()

### Community 20 - "Task Template Generation"
Cohesion: 0.47
Nodes (5): create_problem_class_from_file(), create_scene_xml_file(), main(), This is a script for creating various files frrom templates. This is to ease…, This is just an example for you to jump start. For more advanced editing, you…

### Community 21 - "MimicGen Force Videos"
Cohesion: 0.53
Nodes (5): main(), parse_args(), Namespace, Path, render_task_video()

### Community 22 - "Environment Contact Sheets"
Cohesion: 0.53
Nodes (5): Image, contact_sheet(), labeled(), main(), ndarray

### Community 23 - "Harmonic Sweep Analysis"
Cohesion: 0.60
Nodes (4): process_sequence_harmonic(), Zeroes out all harmonics except the lowest K bands., run_experiment(), spectral_filter_bands()

### Community 24 - "Wavelet Hammering Analysis"
Cohesion: 0.60
Nodes (4): check_checkpoint(), Keeps everything UNFILTERED until the grasp_idx. After the grasp, applies…, run_wavelet_experiment(), wavelet_filter_hybrid()

### Community 25 - "USD Mesh Extraction"
Cohesion: 0.70
Nodes (4): array_body(), extract(), main(), Path

### Community 26 - "HDF5 Inspection"
Cohesion: 0.70
Nodes (4): dataset_rows(), format_attrs(), inspect_file(), main()

### Community 27 - "Action Distribution Analysis"
Cohesion: 0.70
Nodes (4): analyze(), compute_action_kl(), compute_freq_kl(), extract_chunks()

### Community 28 - "UniVTAC Validation"
Cohesion: 0.70
Nodes (4): main(), Group, restore_check(), validate_demo()

### Community 29 - "Trajectory Dashboard"
Cohesion: 0.70
Nodes (4): generate_dashboard(), get_trajectory(), process_full_sequence(), spectral_filter()

### Community 30 - "Spectral Plot Generation"
Cohesion: 0.83
Nodes (3): generate_task_plots(), process_full_sequence(), spectral_filter()

### Community 32 - "Harmonic Dashboard"
Cohesion: 0.83
Nodes (3): generate_dashboard(), process_sequence_harmonic(), spectral_filter_bands()

### Community 33 - "Isolated Shake Dashboard"
Cohesion: 0.83
Nodes (3): generate_isolated_dashboard(), process_hybrid_sequence(), spectral_filter_bands()

## Knowledge Gaps
- **9 isolated node(s):** `Shaken Ketchup Goal`, `IsGrasped Checkpoint`, `Panda Robot`, `OSC Pose Controller`, `Visual Observations` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `XboxControllerHID` connect `Xbox Controller Interface` to `Demonstration Collection`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **What connects `Shaken Ketchup Goal`, `IsGrasped Checkpoint`, `Panda Robot` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ManiSkill MuJoCo Retargeting` be split into smaller, more focused modules?**
  _Cohesion score 0.1411764705882353 - nodes in this community are weakly interconnected._
- **Should `Demonstration Collection` be split into smaller, more focused modules?**
  _Cohesion score 0.09230769230769231 - nodes in this community are weakly interconnected._
- **Should `Xbox Controller Interface` be split into smaller, more focused modules?**
  _Cohesion score 0.13970588235294118 - nodes in this community are weakly interconnected._
- **Should `Demonstration Data Model` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._