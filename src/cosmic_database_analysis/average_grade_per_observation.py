from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy
from sqlalchemy import func
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

VLASS_VERSIONS = ['3.1','3.2']  # List of VLASS versions to process

# Database setup
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)
metadata = sqlalchemy.MetaData()
metadata.reflect(bind=cosmicdb_engine.engine)
cutoff_date = dt.strptime('2023-03-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Define histogram bin edges
bins = [i * 0.1 for i in range(11)]

# Create subplots
fig, axes = plt.subplots(len(VLASS_VERSIONS), 1, figsize=(6, 4 * len(VLASS_VERSIONS)), sharex=True, tight_layout=True)

if len(VLASS_VERSIONS) == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for idx, VLASS_VERSION in enumerate(VLASS_VERSIONS):
    with cosmicdb_engine.session() as session:
        # Query all calibration scans
        query = (
            sqlalchemy.select(
                entities.CosmicDB_Observation.id,
                entities.CosmicDB_Observation.scan_id,
                entities.CosmicDB_Observation.start,
                entities.CosmicDB_ObservationCalibration.overall_grade
            )
            .join(
                entities.CosmicDB_ObservationCalibration,
                entities.CosmicDB_ObservationCalibration.observation_id == entities.CosmicDB_Observation.id
            )
            .where(
                entities.CosmicDB_Observation.scan_id.like(f'%VLASS{VLASS_VERSION}%'),
                entities.CosmicDB_Observation.start > cutoff_date
            )
        )

        results = session.execute(query).fetchall()

        # Organize scans by dataset
        dataset_scans = {}
        dataset_first_scans = {}

        for obs_id, scan_id, start_time, grade in results:
            dataset_id = '.'.join(scan_id.split('.')[:-2])  # Extract dataset ID
            scan_number = int(scan_id.split('.')[-2])  # Extract scan number

            if dataset_id not in dataset_scans:
                dataset_scans[dataset_id] = []

            dataset_scans[dataset_id].append((scan_number, grade))
        print(len(dataset_scans))

        # Identify first scan in each dataset
        for dataset_id, scans in dataset_scans.items():
            first_scan_number = min(scans, key=lambda x: x[0])[0]  # Find the lowest scan number
            dataset_first_scans[dataset_id] = first_scan_number  # Store first scan number

        # Compute average grade per dataset (excluding first scan)
        dataset_avg_grades = {}
        for dataset_id, scans in dataset_scans.items():
            filtered_grades = [grade for scan_number, grade in scans if scan_number != dataset_first_scans[dataset_id]]
            
            if filtered_grades:  # Ensure there's data to compute an average
                dataset_avg_grades[dataset_id] = np.mean(filtered_grades)

        # Extract grades for plotting
        average_grades = list(dataset_avg_grades.values())

        # Plot histogram
        if average_grades:
            axes[idx].hist(average_grades, bins=bins, edgecolor='black')
            axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')
            if idx == len(VLASS_VERSIONS) - 1:
                axes[idx].set_xlabel('Average Calibration Grade (Excluding First Scan)')
        else:
            axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, transform=axes[idx].transAxes)

# Save plot
plt.savefig('average_calibration_grade_excluding_first_scan.png', dpi=300)
