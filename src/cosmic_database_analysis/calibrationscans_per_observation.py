import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from cosmic_database.engine import CosmicDB_Engine
# Import strptime from datetime
from datetime import datetime as dt
import matplotlib.pyplot as plt

VLASS_VERSION = '3.2'

"""
BASIC REQUIRED STUFF TO GET DATABASE ACCESS
"""
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)

# Create a MetaData instance
metadata = sqlalchemy.MetaData()

# Reflect the tables
metadata.reflect(bind=cosmicdb_engine.engine)

# Access tables
CosmicObservation = metadata.tables['cosmic_observation']
CosmicObservationCalibration = metadata.tables['cosmic_observation_calibration']

# Create a session
Session = sessionmaker(bind=cosmicdb_engine.engine)
session = Session()

# Define the cutoff date
cutoff_date = dt.strptime('2023-03-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Query to find all entries in cosmic_observation where scan_id contains "VLASS3.1" and start date is after March 30th, 2023
vlass_scans_query = select(CosmicObservation.c.id, CosmicObservation.c.scan_id).where(
    CosmicObservation.c.scan_id.like(f'%VLASS{VLASS_VERSION}%'),
    CosmicObservation.c.start > cutoff_date
)
# Execute the query
vlass_scans = session.execute(vlass_scans_query).fetchall()

# Dictionary to store the counts of scans per observation group
scan_counts_per_observation = {}
calibration_counts_per_observation = {}

# Iterate over the results and group by observation ID prefix
for scan in vlass_scans:
    # Extract the observation ID prefix (everything except the last two segments)
    observation_id_prefix = '.'.join(scan.scan_id.split('.')[:-2])
    if observation_id_prefix not in scan_counts_per_observation:
        scan_counts_per_observation[observation_id_prefix] = 0
    scan_counts_per_observation[observation_id_prefix] += 1

    # Check if the id exists in cosmic_observation_calibration
    calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == scan.id)
    calibration_entry = session.execute(calibration_query).fetchone()
    
    # Update the calibration counts dictionary
    if observation_id_prefix not in calibration_counts_per_observation:
        calibration_counts_per_observation[observation_id_prefix] = 0
    if calibration_entry:
        calibration_counts_per_observation[observation_id_prefix] += 1

session.close()

vlass_percent_calibrations = {}
# Print the results
print("Counts of scans per observation group:")
for observation_id_prefix, count in scan_counts_per_observation.items():
    print(f"Observation ID Prefix: {observation_id_prefix}, Count: {count}")
    if observation_id_prefix in calibration_counts_per_observation:
        print(f"Calibration Count: {calibration_counts_per_observation[observation_id_prefix]}")
        vlass_percent_calibrations[observation_id_prefix] = calibration_counts_per_observation[observation_id_prefix] / count
    else:
        continue

print("\nCounts of calibration scans per observation group:")
for observation_id_prefix, count in calibration_counts_per_observation.items():
    print(f"Observation ID Prefix: {observation_id_prefix}, Calibration Count: {count}")


print(vlass_percent_calibrations)
# Close the session

# Extract the observation IDs and ratios
observation_ids = list(vlass_percent_calibrations.keys())
ratios = list(vlass_percent_calibrations.values())

# Create the plot
plt.figure(figsize=(15, 12))
plt.barh(observation_ids, ratios, color='skyblue')
plt.xlabel('Ratio of Calibration Observations to Overall Observations')
plt.ylabel('Observation ID')
plt.title('Calibration Observations Ratio per Observation ID')
plt.tight_layout()

# Show the plot
plt.savefig(f'calibration_percentage_per_obs_{VLASS_VERSION}.png')