from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
from sqlalchemy import create_engine, func, select
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import json
import math
from datetime import datetime as dt
from collections import defaultdict
import matplotlib.pyplot as plt

VLASS_VERSION = '3.2'

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
observations_query = select(CosmicObservation).where(
    CosmicObservation.c.scan_id.like(f"VLASS{VLASS_VERSION}%"),
    CosmicObservation.c.start > cutoff_date
)
observations = session.execute(observations_query).fetchall()

# Group observations by their prefix (strip off the last two digits)
prefix_observations = defaultdict(list)
for observation in observations:
    prefix = observation.scan_id.rsplit('.', 2)[0]
    prefix_observations[prefix].append(observation)

separation_per_dataset = {}

# Function to get RA and DEC for a given scan_id
def get_ra_dec(scan_id):
    scan_query = select(CosmicScan).where(CosmicScan.c.id == scan_id)
    scan_entry = session.execute(scan_query).fetchone()
    if scan_entry:
        metadata = json.loads(scan_entry.metadata_json)
        if metadata is not None:
            if "CALIBRATE_FLUX" not in metadata["intents"]["ScanIntent"]:
                ra = metadata['ra_deg']
                dec = metadata['dec_deg']
                return ra, dec
    return None, None

# Iterate through each prefix and calculate separations between calibrator and non-calibrator scans
for prefix, observations in prefix_observations.items():
    separations = []

    for i in range(len(observations) - 1):
        current_scan_id = observations[i].scan_id
        next_scan_id = observations[i + 1].scan_id

        # Check if current scan is a calibrator and next scan is a non-calibrator
        current_calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == observations[i].id)
        next_calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == observations[i + 1].id)
        
        current_calibration_entries = session.execute(current_calibration_query).fetchall()
        next_calibration_entries = session.execute(next_calibration_query).fetchall()
        
        is_current_calibrator = bool(current_calibration_entries)
        is_next_non_calibrator = not bool(next_calibration_entries)

        if is_current_calibrator and is_next_non_calibrator:
            ra_current, dec_current = get_ra_dec(current_scan_id)
            ra_next, dec_next = get_ra_dec(next_scan_id)
            if ra_current is not None and dec_current is not None and ra_next is not None and dec_next is not None:
                print(f"Measuring separation between {current_scan_id} and {next_scan_id}")
                print(f"RA Calibrator: {ra_current}, DEC Calibrator: {dec_current}")
                print(f"RA Target: {ra_next}, DEC Target: {dec_next}")
                separation = math.sqrt((ra_current - ra_next) ** 2 + (dec_current - dec_next) ** 2)
                print(f"Separation: {separation}")
                separations.append(separation)
    
    if separations:
        mean_separation = sum(separations) / len(separations)
        separation_std_dev = math.sqrt(sum((x - mean_separation) ** 2 for x in separations) / len(separations))
        separation_per_dataset[prefix] = {
            "mean_separation": mean_separation,
            "separation_std_dev": separation_std_dev
        }
    else:
        separation_per_dataset[prefix] = {
            "mean_separation": None,
            "separation_std_dev": None
        }

    # print(f"Prefix: {prefix}")
    # print(f"Mean Separation: {separation_per_dataset[prefix]['mean_separation']}")
    # print(f"Separation Std Dev: {separation_per_dataset[prefix]['separation_std_dev']}")
    # print("--------------------------------------------------")

# Plotting the results
observation_ids = list(separation_per_dataset.keys())
mean_separations = [data["mean_separation"] if data["mean_separation"] is not None else -1 for data in separation_per_dataset.values()]
separation_std_devs = [data["separation_std_dev"] if data["separation_std_dev"] is not None else 0 for data in separation_per_dataset.values()]
plt.figure(figsize=(15, 12))
plt.barh(observation_ids, mean_separations, xerr=separation_std_devs, color='skyblue', capsize=5)
plt.xlabel('Mean Separation (degrees)')
plt.ylabel('Observation ID')
plt.title(f'Separation between Calibration and Non-Calibration Observations for VLASS{VLASS_VERSION}')
plt.tight_layout()
plt.savefig(f'separation_between_calibration_and_non_calibration_observations_updated_{VLASS_VERSION}.png')
