from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy
from sqlalchemy import func
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

VLASS_VERSIONS = ['3.1']  # List of VLASS versions to process

# Basic database setup
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)
metadata = sqlalchemy.MetaData()
metadata.reflect(bind=cosmicdb_engine.engine)
cutoff_date = dt.strptime('2023-03-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Create subplots
fig, axes = plt.subplots(len(VLASS_VERSIONS), 1, figsize=(6, 4 * len(VLASS_VERSIONS)), sharex=True, tight_layout=True)

if len(VLASS_VERSIONS) == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for idx, VLASS_VERSION in enumerate(VLASS_VERSIONS):
    with cosmicdb_engine.session() as session:
        # Step 1: Find all observations for the current VLASS_VERSION
        observations_query = (
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
        observations = session.execute(observations_query).fetchall()

        # Group observations by dataset
        datasets = {}
        for obs in observations:
            dataset_id = '.'.join(obs.scan_id.split('.')[:-2]) # Extract dataset ID
            if dataset_id not in datasets:
                datasets[dataset_id] = []
            datasets[dataset_id].append(obs)

        # Step 2: Check for calibration observations
        calibration_counts = []
        print(len(datasets))
        for dataset_id, scans in datasets.items():
            # Sort scans by start time
            scans = sorted(scans, key=lambda x: x.start)

            # Find calibration scans
            calibration_scans = [
                scan for scan in scans
                if session.execute(
                    sqlalchemy.select(entities.CosmicDB_ObservationCalibration.overall_grade)
                    .where(entities.CosmicDB_ObservationCalibration.observation_id == scan.id)
                ).fetchone()
            ]

            # Step 3: Count calibration scans until grade > 0.6
            count = 0
            for cal_scan in calibration_scans:
                grade = session.execute(
                    sqlalchemy.select(entities.CosmicDB_ObservationCalibration.overall_grade)
                    .where(entities.CosmicDB_ObservationCalibration.observation_id == cal_scan.id)
                ).scalar()
                if grade > 0.6:
                    break
                count += 1

            calibration_counts.append(count)

        # Step 4: Plot the histogram
        if not calibration_counts:
            print(f"No data for VLASS {VLASS_VERSION}. Skipping plot.")
            axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, transform=axes[idx].transAxes)
            axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')
            if idx == len(VLASS_VERSIONS) - 1:
                axes[idx].set_xlabel('Calibration Scans Until Grade > 0.6')
            continue

        # Ensure the 0 bin is included
        if 0 not in calibration_counts:
            calibration_counts.append(0)

        axes[idx].hist(calibration_counts, bins=range(0, max(calibration_counts) + 2), edgecolor='black', align='left')
        axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')
        if idx == len(VLASS_VERSIONS) - 1:
            axes[idx].set_xlabel('Calibration Scans Until Grade > 0.6')

# Save the plot
plt.savefig('calibration_scans_until_grade_above_0_6.png', dpi=300)