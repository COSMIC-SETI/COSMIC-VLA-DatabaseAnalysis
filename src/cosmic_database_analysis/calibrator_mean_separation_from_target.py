from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
from sqlalchemy import create_engine, func, select
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import json
import math
from collections import defaultdict
from datetime import datetime as dt
import matplotlib.pyplot as plt

VLASS_VERSION = '3.1'

# BASIC REQUIRED STUFF TO GET DATABASE ACCESS
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)

# Create a MetaData instance
metadata = sqlalchemy.MetaData()

# Reflect the tables
metadata.reflect(bind=cosmicdb_engine.engine)

# Create a session
Session = sessionmaker(bind=cosmicdb_engine.engine)
session = Session()

# Access tables
CosmicObservation = metadata.tables['cosmic_observation']
CosmicObservationCalibration = metadata.tables['cosmic_observation_calibration']
CosmicScan = metadata.tables['cosmic_scan']

# Define the cutoff date
cutoff_date = dt.strptime('2023-03-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Step 1: Find all observations with 'VLASS3.1' in their scan_id
observations_query = select(CosmicObservation).where(CosmicObservation.c.scan_id.like(f"VLASS{VLASS_VERSION}%"),
    CosmicObservation.c.start > cutoff_date)
observations = session.execute(observations_query).fetchall()

# Group observations by their prefix (strip off the last two digits)
prefix_observations = defaultdict(list)
for observation in observations:
    prefix = observation.scan_id.rsplit('.', 2)[0]
    prefix_observations[prefix].append(observation)

separation_per_dataset = {}

# Iterate through each prefix and calculate means and separations
for prefix, observations in prefix_observations.items():
    # Step 2: Check if each observation ID is in cosmic_observation_calibration and find corresponding scan ids
    calibration_scan_ids = []
    for observation in observations:
        calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == observation.id)
        calibration_entries = session.execute(calibration_query).fetchall()

        if calibration_entries:
            calibration_scan_ids.append(observation.scan_id)

    # Step 3: Find the RA and DEC for each calibration scan
    ra_values_calibration = []
    dec_values_calibration = []
    for scan_id in calibration_scan_ids:
        scan_query = select(CosmicScan).where(CosmicScan.c.id == scan_id)
        scan_entry = session.execute(scan_query).fetchone()

        if scan_entry:
            metadata = json.loads(scan_entry.metadata_json)
            if metadata is not None:
                if "CALIBRATE_FLUX" not in metadata["intents"]["ScanIntent"] and "CALIBRATE_BANDPASS" not in metadata["intents"]["ScanIntent"]:
                    print(metadata["intents"]["ScanIntent"])
                    ra_values_calibration.append(metadata['ra_deg'])
                    dec_values_calibration.append(metadata['dec_deg'])
                else:
                    ra_values_calibration.append(None)
                    dec_values_calibration.append(None)
            else:
                ra_values_calibration.append(None)
                dec_values_calibration.append(None)

    # # Print RA and DEC for all calibration scans
    # print(f"RA and DEC for all calibration scans (Prefix: {prefix}):")
    # for ra, dec in zip(ra_values_calibration, dec_values_calibration):
    #     print(f"RA: {ra}, DEC: {dec}")

    # Step 4: Calculate the mean and standard deviation for RA and DEC of calibration scans
    ra_values_calibration_filtered = [x for x in ra_values_calibration if x is not None]
    dec_values_calibration_filtered = [x for x in dec_values_calibration if x is not None]

    mean_ra_calibration = sum(ra_values_calibration_filtered) / len(ra_values_calibration_filtered) if ra_values_calibration_filtered else None
    mean_dec_calibration = sum(dec_values_calibration_filtered) / len(dec_values_calibration_filtered) if dec_values_calibration_filtered else None
    std_dev_ra_calibration = math.sqrt(sum((x - mean_ra_calibration) ** 2 for x in ra_values_calibration_filtered) / len(ra_values_calibration_filtered)) if ra_values_calibration_filtered else None
    std_dev_dec_calibration = math.sqrt(sum((x - mean_dec_calibration) ** 2 for x in dec_values_calibration_filtered) / len(dec_values_calibration_filtered)) if dec_values_calibration_filtered else None

    print(f"Mean RA (Calibration) for {prefix}: {mean_ra_calibration}, Std Dev RA (Calibration): {std_dev_ra_calibration}")
    print(f"Mean DEC (Calibration) for {prefix}: {mean_dec_calibration}, Std Dev DEC (Calibration): {std_dev_dec_calibration}")

    # Step 5: Find the RA and DEC for each non-calibration scan
    non_calibration_scan_ids = set(observation.scan_id for observation in observations) - set(calibration_scan_ids)
    ra_values_non_calibration = []
    dec_values_non_calibration = []
    for scan_id in non_calibration_scan_ids:
        scan_query = select(CosmicScan).where(CosmicScan.c.id == scan_id)
        scan_entry = session.execute(scan_query).fetchone()

        if scan_entry:
            metadata = json.loads(scan_entry.metadata_json)
            if metadata is not None:
                ra_values_non_calibration.append(metadata['ra_deg'])
                dec_values_non_calibration.append(metadata['dec_deg'])
            else:
                ra_values_non_calibration.append(None)
                dec_values_non_calibration.append(None)

    # Step 6: Calculate the mean and standard deviation for RA and DEC of non-calibration scans
    ra_values_non_calibration_filtered = [x for x in ra_values_non_calibration if x is not None]
    dec_values_non_calibration_filtered = [x for x in dec_values_non_calibration if x is not None]
    mean_ra_non_calibration = sum(ra_values_non_calibration_filtered) / len(ra_values_non_calibration_filtered) if ra_values_non_calibration_filtered else None
    mean_dec_non_calibration = sum(dec_values_non_calibration_filtered) / len(dec_values_non_calibration_filtered) if dec_values_non_calibration_filtered else None

    std_dev_ra_non_calibration = math.sqrt(sum((x - mean_ra_non_calibration) ** 2 for x in ra_values_non_calibration_filtered) / len(ra_values_non_calibration_filtered)) if ra_values_non_calibration_filtered else None
    std_dev_dec_non_calibration = math.sqrt(sum((x - mean_dec_non_calibration) ** 2 for x in dec_values_non_calibration_filtered) / len(dec_values_non_calibration_filtered)) if dec_values_non_calibration_filtered else None

    print(f"Mean RA (Non-Calibration) for {prefix}: {mean_ra_non_calibration}, Std Dev RA (Non-Calibration): {std_dev_ra_non_calibration}")
    print(f"Mean DEC (Non-Calibration) for {prefix}: {mean_dec_non_calibration}, Std Dev DEC (Non-Calibration): {std_dev_dec_non_calibration}")

    # Step 7: Calculate the separation between the means (calibrator vs non-calibrator) if both means are available
    if mean_ra_calibration is not None and mean_dec_calibration is not None and mean_ra_non_calibration is not None and mean_dec_non_calibration is not None:
        separation = math.sqrt((mean_ra_calibration - mean_ra_non_calibration) ** 2 + (mean_dec_calibration - mean_dec_non_calibration) ** 2)
        print(f"Separation (degrees) for {prefix}: {separation}")

        # Step 8: Calculate the separation error using the std_deviation if all std_devs are available
        if std_dev_ra_calibration is not None and std_dev_dec_calibration is not None and std_dev_ra_non_calibration is not None and std_dev_dec_non_calibration is not None:
            separation_error = math.sqrt(std_dev_ra_calibration ** 2 + std_dev_dec_calibration ** 2 + std_dev_ra_non_calibration ** 2 + std_dev_dec_non_calibration ** 2)
            print(f"Separation Error for {prefix}: {separation_error}")
    else:
        separation = None
        separation_error = None
        print(f"Cannot calculate separation or separation error for {prefix} due to missing data.")
    print("--------------------------------------------------")

    separation_per_dataset[prefix] = {
        "separation": separation,
        "separation_error": separation_error,
        "mean_ra_calibration": mean_ra_calibration,
        "mean_dec_calibration": mean_dec_calibration,
        "mean_ra_non_calibration": mean_ra_non_calibration,
        "mean_dec_non_calibration": mean_dec_non_calibration
    }

# Plotting the results
observation_ids = list(separation_per_dataset.keys())
separations = [data["separation"] if data["separation"] is not None else -1 for data in separation_per_dataset.values()]
separation_errors = [data["separation_error"] if data["separation_error"] is not None else 0 for data in separation_per_dataset.values()]
plt.figure(figsize=(15, 12))
plt.barh(observation_ids, separations, xerr=separation_errors, color='skyblue', capsize=5)
plt.xlabel('Separation (degrees)')
plt.ylabel('Observation ID')
plt.title(f'Separation between Calibration and Non-Calibration Observations for VLASS{VLASS_VERSION}')
plt.tight_layout()
plt.savefig(f'separation_between_calibration_and_non_calibration_observations_new_{VLASS_VERSION}.png')
