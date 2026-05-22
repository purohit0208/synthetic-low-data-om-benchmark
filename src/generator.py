import os
import json
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define forbidden columns for leakage prevention
FORBIDDEN_COLUMNS = [
    'latent_degradation', 'health_score', 'true_risk_score', 
    'fault_probability_true', 'future_fault_count', 
    'future_maintenance_flag', 'post_event_status', 
    'maintenance_done_after_prediction', 'remaining_useful_life_true',
    'degradation', 'true_fault', 'true_fault_type'
]

def audit_features(df):
    """
    Checks if any forbidden columns are present in the feature dataframe.
    Fails loudly if leakage is detected.
    """
    detected = set()
    for col in df.columns:
        for forbidden in FORBIDDEN_COLUMNS:
            if forbidden.lower() in col.lower():
                detected.add(col)
    if detected:
        raise ValueError(f"CRITICAL DATA LEAKAGE ERROR: The following forbidden columns were found in the dataset: {sorted(detected)}")
    print("Leakage audit passed: No forbidden columns detected.")

def generate_synthetic_data(
    num_plants=6,
    assets_per_plant=80,
    days=365,
    seed=42,
    base_report_probability=0.20,
    moderate_degradation_report_probability=0.40,
    high_degradation_report_probability=0.75,
    maintenance_report_probability=0.95,
):
    """
    Simulates a latent degradation process and generates structured features, 
    imperfect technician reports, and target labels.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    plant_ids = [f"Plant_{i+1}" for i in range(num_plants)]
    asset_types = ["Pump", "Compressor", "Conveyor", "Motor", "Gearbox"]
    
    # Define components mapped to asset types
    asset_components = {
        "Pump": ["Bearing", "Seal", "Valve"],
        "Compressor": ["Bearing", "Seal", "Valve", "Cooling Unit"],
        "Conveyor": ["Bearing", "Belt", "Motor Unit"],
        "Motor": ["Bearing", "Motor Unit", "Sensor"],
        "Gearbox": ["Bearing", "Seal", "Pressure Unit"]
    }
    
    # Site operating conditions (scale factors for degradation rate)
    site_degradation_scales = {
        "Plant_1": 1.0,  # Baseline
        "Plant_2": 1.15, # Slightly harsher
        "Plant_3": 0.85, # Milder
        "Plant_4": 1.25, # Harsh
        "Plant_5": 0.90, # Mild
        "Plant_6": 1.40  # Held-out testing site (harsh env)
    }
    
    # Asset definition
    assets = []
    total_assets = num_plants * assets_per_plant
    for i in range(total_assets):
        plant_id = plant_ids[i // assets_per_plant]
        asset_type = np.random.choice(asset_types)
        components = asset_components[asset_type]
        component_type = np.random.choice(components)
        
        # Base parameters for the asset
        assets.append({
            "asset_id": f"AST_{i+1:03d}",
            "plant_id": plant_id,
            "asset_type": asset_type,
            "component_type": component_type,
            "base_degradation_rate": np.random.uniform(0.0008, 0.0018),
            "duty_cycle": int(np.random.choice([8, 12, 16, 24])),
            "base_age": np.random.randint(30, 730), # age in days at simulation start
            "initial_degradation": np.random.uniform(0.0, 0.2),
            "previous_faults": np.random.randint(0, 3)
        })
        
    start_date = datetime.date(2025, 1, 1)
    records = []
    
    print(f"Simulating {total_assets} assets over {days} days...")
    
    # Technician report templates by fault type
    text_templates = {
        "Normal": [
            "Routine inspection completed. Asset operating within normal parameters.",
            "Regular checkup. No anomalies detected.",
            "General maintenance inspection. Asset is in good condition.",
            "Asset running smoothly. Temp and vibration normal.",
            "Routine check, normal operation.",
            "Inspection completed. No immediate intervention required.",
            "Asset operating normally after inspection."
        ],
        "Bearing Wear": [
            "abnormal vibration noticed during inspection",
            "intermittent noise under high load, bearings may require replacement",
            "vibration levels elevated, wear suspected on bearing housing",
            "high frequency vibration detected, lubrication applied but noise persists",
            "unusual acoustic pattern near bearing assembly, monitor closely"
        ],
        "Seal Leakage": [
            "seal leakage suspected, minor fluid build-up around casing",
            "leakage detected at primary seal, recommend seal replacement during next shutdown",
            "fluid residue noted, seals showing signs of degradation",
            "inspected seals, minor wear present but seal integrity still holds",
            "visible leakage at gasket seal, pressure drop observed"
        ],
        "Overheating": [
            "minor temperature increase, no immediate action",
            "temperature spike after load increase, cooling unit check advised",
            "thermal scan shows localized hot spot on casing",
            "casing temperature elevated, technician recommended adjusting load factor",
            "overheating detected during high load cycle, thermal safety switch active"
        ],
        "Misalignment": [
            "unusual acoustic pattern but no visible damage",
            "suspected misalignment, shaft vibration slightly off-axis",
            "coupling inspection shows minor wear, possible alignment issue",
            "vibration spectrum indicates potential misalignment under load",
            "technician adjusted mounts to reduce axial vibration deviation"
        ],
        "Blockage": [
            "pressure fluctuation observed, monitor next shift",
            "flow rate deviation exceeds tolerance, potential blockage in valves",
            "suction pressure low, check inlet line for obstruction",
            "differential pressure elevated, filter replacement recommended",
            "minor blockage suspected due to flow rate drop"
        ],
        "Sensor Drift": [
            "sensor readings fluctuating wildly, connection checked",
            "suspected sensor drift, calibration offset updated",
            "vibration sensor reporting erratic signals, replacement scheduled",
            "temperature sensor deviation from thermal scan, calibration needed",
            "spurious readings observed on telemetry channel"
        ]
    }
    
    # Simulate day-by-day
    for day_idx in range(days):
        current_date = start_date + datetime.timedelta(days=day_idx)
        # Seasonal ambient factors
        seasonal_temp_offset = 10.0 * np.sin(2.0 * np.pi * day_idx / 365.0)
        
        for asset in assets:
            # Initialize or retrieve persistent asset-level states
            if "degradation" not in asset:
                asset["degradation"] = asset["initial_degradation"]
                asset["age"] = asset["base_age"]
                asset["operating_hours"] = asset["base_age"] * asset["duty_cycle"]
                asset["time_since_maint"] = np.random.randint(5, 90)
                asset["fault_cooldown"] = 0
                asset["current_fault"] = "Normal"
            
            # Update aging and hours
            asset["age"] += 1
            asset["operating_hours"] += asset["duty_cycle"]
            asset["time_since_maint"] += 1
            
            # Load factor for the day
            load_factor = np.random.uniform(0.5, 1.1)
            # Add site and duty cycle scaling to load factor impact
            site_scale = site_degradation_scales[asset["plant_id"]]
            
            # Degradation increment
            deg_increment = asset["base_degradation_rate"] * load_factor * site_scale * (1.0 + asset["age"] / 1000.0)
            
            # Random shock
            if np.random.rand() < 0.005:
                deg_increment += np.random.uniform(0.08, 0.20)
                
            asset["degradation"] = min(1.2, asset["degradation"] + deg_increment)
            
            # Check for fault triggers
            if asset["degradation"] >= 0.75 and asset["current_fault"] == "Normal" and asset["fault_cooldown"] <= 0:
                # Select fault type based on asset capability
                if asset["asset_type"] == "Pump":
                    asset["current_fault"] = np.random.choice(["Bearing Wear", "Seal Leakage", "Blockage"])
                elif asset["asset_type"] == "Compressor":
                    asset["current_fault"] = np.random.choice(["Bearing Wear", "Overheating", "Seal Leakage"])
                elif asset["asset_type"] == "Conveyor":
                    asset["current_fault"] = np.random.choice(["Bearing Wear", "Misalignment", "Overheating"])
                elif asset["asset_type"] == "Motor":
                    asset["current_fault"] = np.random.choice(["Bearing Wear", "Overheating", "Sensor Drift"])
                else: # Gearbox
                    asset["current_fault"] = np.random.choice(["Bearing Wear", "Seal Leakage", "Misalignment"])
                    
            # Maintenance decision logic (simulation of scheduled / corrective maintenance)
            maint_triggered = False
            # Corrective maintenance if fault is active for too long or degradation is critical
            if asset["degradation"] >= 0.95 and np.random.rand() < 0.25:
                maint_triggered = True
            elif asset["degradation"] >= 0.75 and np.random.rand() < 0.10:
                maint_triggered = True
            # Scheduled maintenance based on time since last maintenance
            elif asset["time_since_maint"] > 180 and np.random.rand() < 0.05:
                maint_triggered = True
                
            if maint_triggered:
                # Reset degradation and fault status
                if asset["current_fault"] != "Normal":
                    asset["previous_faults"] += 1
                asset["degradation"] = np.random.uniform(0.0, 0.08)
                asset["current_fault"] = "Normal"
                asset["time_since_maint"] = 0
                asset["fault_cooldown"] = 15 # Wait 15 days before next fault can trigger
            else:
                if asset["fault_cooldown"] > 0:
                    asset["fault_cooldown"] -= 1
                    
            # Generate sensor features
            # Ambient conditions
            ambient_temp = 20.0 + seasonal_temp_offset + np.random.normal(0, 2)
            # Site offset for environmental conditions
            if asset["plant_id"] == "Plant_4": # Dry and hot site
                ambient_temp += 5.0
                humidity = max(10.0, 40.0 + np.random.normal(0, 5))
            elif asset["plant_id"] == "Plant_3": # Humid and cool site
                ambient_temp -= 3.0
                humidity = min(98.0, 75.0 + np.random.normal(0, 4))
            else:
                humidity = max(10.0, min(95.0, 55.0 + np.random.normal(0, 8)))
                
            # Vibration features increase with degradation and load
            vib_mult = 1.0 + 3.5 * (asset["degradation"] ** 2)
            vibration_rms = max(0.2, float(np.random.normal(1.2 * vib_mult, 0.15 * load_factor)))
            vibration_kurtosis = max(2.0, float(3.0 + 8.0 * (asset["degradation"] ** 3) + np.random.normal(0, 0.4)))
            
            # Acoustic levels increase with degradation
            acoustic_level = max(45.0, float(55.0 + 22.0 * (asset["degradation"] ** 1.5) + np.random.normal(0, 1.8)))
            
            # Motor current increases with load factor and bearing degradation
            motor_current = max(1.0, float(15.0 * load_factor + 8.0 * asset["degradation"] + np.random.normal(0, 0.5)))
            
            # Temperature deviation increases with degradation and load
            temp_dev = max(0.5, float(3.0 + 18.0 * (asset["degradation"] ** 2) * load_factor + np.random.normal(0, 0.8)))
            
            # Pressure and flow rate deviations (specific to fluids - Pumps, Compressors)
            press_dev = float(np.random.normal(0.0, 0.3))
            flow_dev = float(np.random.normal(0.0, 0.4))
            
            if asset["asset_type"] in ["Pump", "Compressor"]:
                if asset["current_fault"] == "Blockage":
                    press_dev = float(np.random.normal(2.5, 0.4))
                    flow_dev = float(np.random.normal(-3.5, 0.5))
                elif asset["current_fault"] == "Seal Leakage":
                    press_dev = float(np.random.normal(-1.8, 0.3))
                    flow_dev = float(np.random.normal(-1.2, 0.4))
                else:
                    press_dev = float(np.random.normal(0.5 * asset["degradation"], 0.3))
                    flow_dev = float(np.random.normal(-0.5 * asset["degradation"], 0.4))
                    
            # Shift type
            shift = np.random.choice(["Day", "Swing", "Night"], p=[0.4, 0.35, 0.25])
            
            # Save raw information for target computation and leakage checks
            records.append({
                "asset_id": asset["asset_id"],
                "site_id": asset["plant_id"],
                "asset_type": asset["asset_type"],
                "component_type": asset["component_type"],
                "date": current_date,
                "age": asset["age"],
                "operating_hours": asset["operating_hours"],
                "load_factor": load_factor,
                "duty_cycle": asset["duty_cycle"],
                "ambient_temp": ambient_temp,
                "humidity": humidity,
                "vibration_rms": vibration_rms,
                "vibration_kurtosis": vibration_kurtosis,
                "acoustic_level": acoustic_level,
                "motor_current": motor_current,
                "temp_deviation": temp_dev,
                "pressure_deviation": press_dev,
                "flow_rate_deviation": flow_dev,
                "time_since_last_maintenance": asset["time_since_maint"],
                "previous_fault_count": asset["previous_faults"],
                "shift_type": shift,
                
                # Hidden variables for target computation
                "_degradation": asset["degradation"],
                "_fault_type": asset["current_fault"],
                "_maintenance_triggered": maint_triggered
            })
            
    df = pd.DataFrame(records)
    
    # Calculate future targets lookahead window = 7 days
    # To avoid lookahead leakage during evaluation, we sort by asset and date
    df = df.sort_values(by=["asset_id", "date"]).reset_index(drop=True)
    
    # targets:
    # 1. maintenance_required_7d: True if there is a maintenance trigger or degradation exceeds 0.75 within next 7 days
    # 2. fault_type: current fault type (Normal, Bearing Wear, Seal Leakage, Overheating, Misalignment, Blockage, Sensor Drift)
    # 3. maintenance_priority: Normal (0), Low (1), Medium (2), High (3), Critical (4)
    # 4. remaining_useful_life_bin: Normal (>30d), 14-30 days, 7-14 days, <7 days
    
    maintenance_required_7d = []
    fault_type = []
    maintenance_priority = []
    rul_bin = []
    
    for idx, row in df.iterrows():
        asset_id = row["asset_id"]
        curr_deg = row["_degradation"]
        curr_fault = row["_fault_type"]
        
        # Get future lookahead records for this asset
        future_rows = df.iloc[idx+1 : idx+8]
        # Filter to keep only the same asset
        future_rows = future_rows[future_rows["asset_id"] == asset_id]
        
        # Target 1: maintenance required if degradation crosses 0.75 or maintenance is triggered in next 7 days
        maint_req = 0
        if curr_deg >= 0.75:
            maint_req = 1
        elif len(future_rows) > 0:
            if future_rows["_degradation"].max() >= 0.75 or future_rows["_maintenance_triggered"].any():
                maint_req = 1
        maintenance_required_7d.append(maint_req)
        
        # Target 2: Multiclass fault type
        fault_type.append(curr_fault)
        
        # Target 3: Maintenance priority
        if curr_deg < 0.3:
            priority = "Normal"
        elif curr_deg < 0.6:
            priority = "Low"
        elif curr_deg < 0.75:
            priority = "Medium"
        elif curr_deg < 0.9:
            priority = "High"
        else:
            priority = "Critical"
        maintenance_priority.append(priority)
        
        # Target 4: RUL binning
        # Find how many days until degradation >= 0.75 (if not already there)
        # Look ahead up to 30 days
        future_30 = df.iloc[idx : idx+31]
        future_30 = future_30[future_30["asset_id"] == asset_id]
        
        if curr_deg >= 0.75:
            rul = "<7 days"
        else:
            # Find first index where degradation >= 0.75
            cross_idx = np.where(future_30["_degradation"].values >= 0.75)[0]
            if len(cross_idx) > 0:
                days_to_fault = cross_idx[0]
                if days_to_fault < 7:
                    rul = "<7 days"
                elif days_to_fault < 14:
                    rul = "7-14 days"
                else:
                    rul = "14-30 days"
            else:
                rul = "Normal"
        rul_bin.append(rul)
        
    df["maintenance_required_7d"] = maintenance_required_7d
    df["fault_type"] = fault_type
    df["maintenance_priority"] = maintenance_priority
    df["remaining_useful_life_bin"] = rul_bin
    
    # Generate imperfect technician reports
    technician_reports = []
    for idx, row in df.iterrows():
        deg = row["_degradation"]
        f_type = row["_fault_type"]
        maint_trig = row["_maintenance_triggered"]
        
        # Decide report presence
        prob = base_report_probability
        if deg >= 0.75:
            prob = high_degradation_report_probability
        elif deg >= 0.4:
            prob = moderate_degradation_report_probability
            
        if maint_trig:
            prob = maintenance_report_probability
            
        if np.random.rand() < prob:
            # Inject noise: 15% probability of generating a mismatched template
            if np.random.rand() < 0.15:
                # Mismatched/noisy report
                avail_types = list(text_templates.keys())
                selected_type = np.random.choice(avail_types)
                report = np.random.choice(text_templates[selected_type])
            else:
                # Match current state
                report = np.random.choice(text_templates[f_type])
                
            # Randomly truncate or make report vague for additional ambiguity
            if np.random.rand() < 0.10:
                report = np.random.choice(["Inspection completed.", "Asset checked.", "Routine log entry."])
        else:
            report = np.nan
            
        technician_reports.append(report)
        
    df["technician_report"] = technician_reports
    
    # Store latent columns separately
    latent_df = df[["asset_id", "date", "_degradation", "_fault_type", "_maintenance_triggered"]].copy()
    latent_df.columns = ["asset_id", "date", "latent_degradation", "latent_fault_type", "latent_maintenance_triggered"]
    
    # Drop hidden columns to create clean model dataframe
    model_df = df.drop(columns=["_degradation", "_fault_type", "_maintenance_triggered"])
    
    generation_config = {
        "num_plants": num_plants,
        "assets_per_plant": assets_per_plant,
        "days": days,
        "seed": seed,
        "start_date": str(start_date),
        "prediction_label": "maintenance_required_7d",
        "prediction_window_days": 7,
        "latent_fault_threshold": 0.75,
        "maintenance_reset_degradation_range": [0.0, 0.08],
        "base_report_probability": base_report_probability,
        "moderate_degradation_report_probability": moderate_degradation_report_probability,
        "high_degradation_report_probability": high_degradation_report_probability,
        "maintenance_report_probability": maintenance_report_probability,
        "site_degradation_scales": site_degradation_scales,
    }
    model_df.attrs["generation_config"] = generation_config
    latent_df.attrs["generation_config"] = generation_config

    return model_df, latent_df

def save_and_split(df, latent_df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    generation_config = df.attrs.get("generation_config", {})
    
    # Run Leakage Audit on model dataframe
    audit_features(df.drop(columns=["maintenance_required_7d", "fault_type", "maintenance_priority", "remaining_useful_life_bin"]))
    
    # Save the main dataset
    df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    df.to_parquet(os.path.join(output_dir, "dataset.parquet"), index=False)
    
    # Save latent database
    latent_df.to_parquet(os.path.join(output_dir, "latent_database.parquet"), index=False)

    # Save reproducibility metadata and generation configuration
    metadata = {
        "generated_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "row_count": int(len(df)),
        "asset_count": int(df["asset_id"].nunique()),
        "site_count": int(df["site_id"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "maintenance_required_7d_positive_rate": float(df["maintenance_required_7d"].mean()),
        "technician_report_coverage": float(df["technician_report"].notna().mean()),
        "fault_type_distribution": {str(k): int(v) for k, v in df["fault_type"].value_counts().items()},
        "maintenance_priority_distribution": {str(k): int(v) for k, v in df["maintenance_priority"].value_counts().items()},
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)
    with open(os.path.join(output_dir, "generation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Save data dictionary
    data_dict = {
        "Column": list(df.columns),
        "Type": [str(t) for t in df.dtypes],
        "Description": [
            "Unique identifier for the physical asset",
            "Plant/Site identifier (Plant_1 to Plant_6)",
            "Type of industrial system (Pump, Compressor, etc.)",
            "Component monitored (Bearing, Seal, Valve, etc.)",
            "Date of simulation record",
            "Asset age in days since installation",
            "Cumulative asset operating hours",
            "Daily load factor (0.5 to 1.1)",
            "Daily shift duration (hours per day)",
            "Ambient temperature in degrees C",
            "Relative humidity percentage",
            "Vibration Root Mean Square (sensor value)",
            "Vibration Kurtosis (sensor value)",
            "Acoustic noise level in dB (sensor value)",
            "Motor current draw in Amperes (sensor value)",
            "Temperature deviation from ambient in degrees C (sensor value)",
            "Pressure deviation in bar (sensor value)",
            "Fluid flow rate deviation in m^3/h (sensor value)",
            "Days since last maintenance intervention",
            "Cumulative count of historical asset faults",
            "Active shift type (Day, Swing, Night)",
            "Predictive binary target (1 if maintenance needed within 7 days, 0 otherwise)",
            "Multiclass fault type (Normal, Bearing Wear, Seal Leakage, etc.)",
            "Ordinal priority classification (Normal, Low, Medium, High, Critical)",
            "Ordinal remaining useful life bin",
            "Short technician-style text notes (imperfect/noisy, can be NaN)"
        ]
    }
    pd.DataFrame(data_dict).to_csv(os.path.join(output_dir, "data_dictionary.csv"), index=False)
    
    # Generate splits
    # Random split: Chronological splitting within each site to avoid future-looking evaluation
    # Train: first 70% of days, Val: next 15% days, Test: last 15% days
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.70)
    val_end = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]
    
    # Standard Split (Plants 1-5 used for training, Plant 6 kept for site-held-out or standard testing depending on experiment)
    # Actually, we can split train/val/test using Plants 1-5, and hold out Plant 6 entirely
    df_plants_1_5 = df[df["site_id"] != "Plant_6"].copy()
    df_plant_6 = df[df["site_id"] == "Plant_6"].copy()
    
    train_split = df_plants_1_5[df_plants_1_5["date"].isin(train_dates)].copy()
    val_split = df_plants_1_5[df_plants_1_5["date"].isin(val_dates)].copy()
    test_split = df_plants_1_5[df_plants_1_5["date"].isin(test_dates)].copy()
    
    # Save standard splits
    train_split.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_split.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_split.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    split_rows = []

    def add_split_manifest(split_name, split_df):
        split_rows.append({
            "split": split_name,
            "row_count": int(len(split_df)),
            "date_min": str(split_df["date"].min()) if len(split_df) else "",
            "date_max": str(split_df["date"].max()) if len(split_df) else "",
            "site_ids": ";".join(sorted(split_df["site_id"].unique())) if len(split_df) else "",
            "positive_rate": float(split_df["maintenance_required_7d"].mean()) if len(split_df) else 0.0,
            "technician_report_coverage": float(split_df["technician_report"].notna().mean()) if len(split_df) else 0.0,
        })

    add_split_manifest("chronological_train_plants_1_5", train_split)
    add_split_manifest("chronological_val_plants_1_5", val_split)
    add_split_manifest("chronological_test_plants_1_5", test_split)

    # Save stratified random split for random-vs-site-held-out comparison.
    random_dir = os.path.join(output_dir, "random_splits")
    os.makedirs(random_dir, exist_ok=True)
    random_train, random_temp = train_test_split(
        df,
        train_size=0.70,
        random_state=generation_config.get("seed", 42),
        stratify=df["maintenance_required_7d"],
    )
    random_val, random_test = train_test_split(
        random_temp,
        train_size=0.50,
        random_state=generation_config.get("seed", 42),
        stratify=random_temp["maintenance_required_7d"],
    )
    random_train.to_parquet(os.path.join(random_dir, "train.parquet"), index=False)
    random_val.to_parquet(os.path.join(random_dir, "val.parquet"), index=False)
    random_test.to_parquet(os.path.join(random_dir, "test.parquet"), index=False)
    add_split_manifest("random_train", random_train)
    add_split_manifest("random_val", random_val)
    add_split_manifest("random_test", random_test)

    # Save fixed label-scarcity training splits.
    scarcity_dir = os.path.join(output_dir, "label_scarcity_splits")
    os.makedirs(scarcity_dir, exist_ok=True)
    label_fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    seed = generation_config.get("seed", 42)
    scarcity_rows = []
    for frac in label_fractions:
        if frac < 1.0:
            scarce_split, _ = train_test_split(
                train_split,
                train_size=frac,
                random_state=seed,
                stratify=train_split["maintenance_required_7d"],
            )
        else:
            scarce_split = train_split.copy()
        split_name = f"train_{int(frac * 100):03d}pct"
        scarce_split.to_parquet(os.path.join(scarcity_dir, f"{split_name}.parquet"), index=False)
        scarcity_rows.append({
            "split": split_name,
            "fraction": frac,
            "row_count": int(len(scarce_split)),
            "positive_count": int(scarce_split["maintenance_required_7d"].sum()),
            "positive_rate": float(scarce_split["maintenance_required_7d"].mean()),
        })
    pd.DataFrame(scarcity_rows).to_csv(os.path.join(output_dir, "label_scarcity_manifest.csv"), index=False)
    
    # Save site-held-out splits for Experiment 2
    # Hold out each plant one by one
    for i in range(1, 7):
        held_out_site = f"Plant_{i}"
        df_train_val = df[df["site_id"] != held_out_site].copy()
        df_test = df[df["site_id"] == held_out_site].copy()
        
        # Split train_val into train (80%) and val (20%) chronologically
        train_val_dates = sorted(df_train_val["date"].unique())
        split_pt = int(len(train_val_dates) * 0.80)
        t_dates = train_val_dates[:split_pt]
        v_dates = train_val_dates[split_pt:]
        
        tr = df_train_val[df_train_val["date"].isin(t_dates)]
        va = df_train_val[df_train_val["date"].isin(v_dates)]
        
        split_dir = os.path.join(output_dir, "site_splits", held_out_site)
        os.makedirs(split_dir, exist_ok=True)
        tr.to_parquet(os.path.join(split_dir, "train.parquet"), index=False)
        va.to_parquet(os.path.join(split_dir, "val.parquet"), index=False)
        df_test.to_parquet(os.path.join(split_dir, "test.parquet"), index=False)
        add_split_manifest(f"site_held_out_{held_out_site}_train", tr)
        add_split_manifest(f"site_held_out_{held_out_site}_val", va)
        add_split_manifest(f"site_held_out_{held_out_site}_test", df_test)

    pd.DataFrame(split_rows).to_csv(os.path.join(output_dir, "split_manifest.csv"), index=False)
        
    print(f"Data saved and split successfully in '{output_dir}'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-leakage", action="store_true")
    args = parser.parse_args()
    
    if args.test_leakage:
        print("Running Leakage Audit verification test...")
        leakage_test_df = pd.DataFrame({
            "age": [100, 200],
            "vibration_rms": [1.2, 1.5],
            "latent_degradation": [0.2, 0.4] # Leakage column
        })
        try:
            audit_features(leakage_test_df)
        except ValueError as e:
            print(f"Test successfully caught leakage: {e}")
            exit(0)
        print("ERROR: Leakage audit failed to catch forbidden column!")
        exit(1)
        
    df, latent_df = generate_synthetic_data(days=args.days, seed=args.seed)
    save_and_split(df, latent_df)
