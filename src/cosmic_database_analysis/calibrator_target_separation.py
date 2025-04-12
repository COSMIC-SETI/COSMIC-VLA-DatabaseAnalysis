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

# Update the function to extract "src" key from metadata
def get_src_ra_dec(scan_id):
    scan_query = select(CosmicScan).where(CosmicScan.c.id == scan_id)
    scan_entry = session.execute(scan_query).fetchone()
    if scan_entry:
        metadata = json.loads(scan_entry.metadata_json)
        if metadata is not None:
            src = metadata.get('src', None)
            ra = metadata['ra_deg']
            dec = metadata['dec_deg']
            return src, ra, dec
    return None, None, None

# Create a dictionary to group by calibrator name
calibrator_grouped_data = defaultdict(list)

# Update the loop to use "src" as the grouping key
for prefix, observations in prefix_observations.items():
    for i in range(len(observations) - 1):
        current_scan_id = observations[i].scan_id
        next_scan_id = observations[i + 1].scan_id

        current_calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == observations[i].id)
        next_calibration_query = select(CosmicObservationCalibration).where(CosmicObservationCalibration.c.observation_id == observations[i + 1].id)

        current_calibration_entries = session.execute(current_calibration_query).fetchall()
        next_calibration_entries = session.execute(next_calibration_query).fetchall()

        is_current_calibrator = bool(current_calibration_entries)
        is_next_non_calibrator = not bool(next_calibration_entries)

        if is_current_calibrator and is_next_non_calibrator:
            src_current, ra_current, dec_current = get_src_ra_dec(current_scan_id)
            _, ra_next, dec_next = get_src_ra_dec(next_scan_id)

            if src_current and ra_current is not None and dec_current is not None and ra_next is not None and dec_next is not None:
                separation = math.sqrt((ra_current - ra_next) ** 2 + (dec_current - dec_next) ** 2)
                calibrator_grouped_data[src_current].append(separation)

# Calculate mean and standard deviation for each calibrator
calibrator_separation_stats = {
    calibrator: {
        "mean_separation": sum(separations) / len(separations),
        "separation_std_dev": math.sqrt(sum((x - (sum(separations) / len(separations))) ** 2 for x in separations) / len(separations))
    } for calibrator, separations in calibrator_grouped_data.items() if separations
}

# Plotting results grouped by calibrator names
calibrator_names = list(calibrator_separation_stats.keys())
mean_separations = [data["mean_separation"] for data in calibrator_separation_stats.values()]
separation_std_devs = [data["separation_std_dev"] for data in calibrator_separation_stats.values()]

plt.figure(figsize=(15, 12))
plt.barh(calibrator_names, mean_separations, xerr=separation_std_devs, color='skyblue', capsize=5)
plt.xlabel('Mean Separation (degrees)')
plt.ylabel('Calibrator Name')
plt.title(f'Separation between Calibrator and Following Target for VLASS{VLASS_VERSION}')
plt.tight_layout()
plt.savefig(f'separation_by_calibrator_{VLASS_VERSION}.png')

# Print out the list of all calibrator names
print("List of all calibrator names used:")
for calibrator in calibrator_names:
    print(calibrator)

# # Function to get the 'overall_grade' of a calibration observation
# def get_overall_grade(observation_id):
#     grade_query = select(CosmicObservationCalibration.c.overall_grade).where(
#         CosmicObservationCalibration.c.observation_id == observation_id
#     )
#     grade_entry = session.execute(grade_query).fetchone()
#     return grade_entry[0] if grade_entry else None

# # Function to check if 'src' contains the word "Slew"
# def contains_slew(scan_id):
#     src, _, _ = get_src_ra_dec(scan_id)
#     if "Slew" in src:
#         print(src)
#         print(src and "Slew" in src)
#     return src and "Slew" in src

# # Iterate over observations to find scans containing "Slew" in their src and their subsequent calibration
# slew_calibrations_and_grades = []

# # Iterate over observations to find scans containing "Slew" in their src and their subsequent calibration
# slew_calibrations_and_grades = []

# # Updated loop to iterate until the next calibration scan is found
# slew_calibrations_and_grades = []

# for prefix, observations in prefix_observations.items():
#     for i in range(len(observations) - 1):
#         current_scan_id = observations[i].scan_id

#         # Check if the current scan's src contains "Slew"
#         if contains_slew(current_scan_id):
#             print(f"Found Slew scan: {current_scan_id}")

#             # Look for the next calibration scan
#             for j in range(i + 1, len(observations)):
#                 next_scan_id = observations[j].scan_id

#                 next_calibration_query = select(CosmicObservationCalibration).where(
#                     CosmicObservationCalibration.c.observation_id == observations[j].id
#                 )
#                 next_calibration_entries = session.execute(next_calibration_query).fetchall()

#                 if next_calibration_entries:
#                     overall_grade = get_overall_grade(observations[j].id)
#                     print(f"Overall grade for {next_scan_id}: {overall_grade}")
#                     slew_calibrations_and_grades.append((current_scan_id, next_scan_id, overall_grade))
#                     break  # Stop once the next calibration scan is found
#             else:
#                 # If no calibration scan is found after the Slew scan
#                 print(f"No subsequent calibration scan found for Slew scan: {current_scan_id}")

# # Print the results
# print("Slew Calibrations and Grades of Following Observations:")
# for slew_scan, next_calibration, grade in slew_calibrations_and_grades:
#     print(f"Slew Scan: {slew_scan}, Next Calibration: {next_calibration}, Grade: {grade}")
