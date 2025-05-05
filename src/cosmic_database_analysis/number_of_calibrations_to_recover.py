from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

VLASS_VERSIONS = ['3.1','3.2']  # List of VLASS versions to process

# Database setup
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)
cutoff_date = dt.strptime('2023-03-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Create subplots
fig, axes = plt.subplots(len(VLASS_VERSIONS), 1, figsize=(6, 4 * len(VLASS_VERSIONS)), sharex=True, tight_layout=True)
if len(VLASS_VERSIONS) == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for idx, VLASS_VERSION in enumerate(VLASS_VERSIONS):
    with cosmicdb_engine.session() as session:
        # Step 1: Query all calibration observations within VLASS_VERSION
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
        observations = session.execute(query).fetchall()

        # Correct dataset extraction using a set to avoid over-counting
        unique_datasets = {}
        for obs in observations:
            dataset_id = '.'.join(obs.scan_id.split('.')[:-2])
            if dataset_id not in unique_datasets:
                unique_datasets[dataset_id] = []
            unique_datasets[dataset_id].append(obs)

        # Verification print statements
        total_calibration_observations = len(observations)
        total_unique_datasets = len(unique_datasets)
        print(f"VLASS {VLASS_VERSION}: Total calibration observations found: {total_calibration_observations}")
        print(f"VLASS {VLASS_VERSION}: Total unique datasets found: {total_unique_datasets}")

        # Step 2: Count calibration scans per dataset
        calibration_counts = []
        for dataset_id, scans in unique_datasets.items():
            scans = sorted(scans, key=lambda x: x.start)

            # Find calibration scans and count until grade > 0.6
            count = 0
            for scan in scans:
                grade = scan.overall_grade
                if grade > 0.6:
                    break
                count += 1

            calibration_counts.append(count)

        # Step 3: Plot the histogram
        if not calibration_counts:
            print(f"No data for VLASS {VLASS_VERSION}. Skipping plot.")
            axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, transform=axes[idx].transAxes)
            axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')
            if idx == len(VLASS_VERSIONS) - 1:
                axes[idx].set_xlabel('Calibration Scans Until Grade > 0.6')
            continue

        # Ensure 0 bin is included
        if 0 not in calibration_counts:
            calibration_counts.append(0)

        # Plot the histogram
        axes[idx].hist(calibration_counts, bins=range(0, max(calibration_counts) + 2), edgecolor='black', align='left')
        axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')

        # Set x-ticks to whole numbers only
        axes[idx].set_xticks(range(0, max(calibration_counts) + 1))

        if idx == len(VLASS_VERSIONS) - 1:
            axes[idx].set_xlabel('Calibration Scans Until Grade > 0.6')

# Save the plot
plt.savefig('calibration_scans_until_grade_above_0_6.png', dpi=300)