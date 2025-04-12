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

# Define the bin edges for the histogram
bins = [i * 0.1 for i in range(11)]

# Create subplots
fig, axes = plt.subplots(len(VLASS_VERSIONS), 1, figsize=(6, 4 * len(VLASS_VERSIONS)), sharex=True, tight_layout=True)

if len(VLASS_VERSIONS) == 1:
    axes = [axes]  # Ensure axes is iterable for a single subplot

for idx, VLASS_VERSION in enumerate(VLASS_VERSIONS):
    # Extract observation ID from scan_id
    observation_id_expr = func.substr(entities.CosmicDB_Observation.scan_id, 1, func.length(entities.CosmicDB_Observation.scan_id) - 6)
    scan_number_expr = func.substr(entities.CosmicDB_Observation.scan_id, -5, 3)

    # Subquery to find the first calibration scan in each observation
    first_calibration_subquery = (
        sqlalchemy.select(
            observation_id_expr.label('observation_id'),
            func.min(entities.CosmicDB_Observation.start).label('first_calibration_time')
        )
        .join(
            entities.CosmicDB_ObservationCalibration,
            entities.CosmicDB_Observation.id == entities.CosmicDB_ObservationCalibration.observation_id
        )
        .where(entities.CosmicDB_Observation.scan_id.like(f'%VLASS{VLASS_VERSION}%'),
               entities.CosmicDB_Observation.start > cutoff_date)
        .group_by(observation_id_expr)
        .subquery()
    )

    # Query to count calibration scans until grade > 0.6
    query = (
        sqlalchemy.select(
            observation_id_expr.label('observation_id'),
            func.count(entities.CosmicDB_ObservationCalibration.overall_grade).label('calibration_count')
        )
        .join(
            entities.CosmicDB_Observation,
            entities.CosmicDB_ObservationCalibration.observation_id == entities.CosmicDB_Observation.id
        )
        .join(
            first_calibration_subquery,
            (observation_id_expr == first_calibration_subquery.c.observation_id) &
            (entities.CosmicDB_Observation.start >= first_calibration_subquery.c.first_calibration_time)
        )
        .where(entities.CosmicDB_Observation.scan_id.like(f'%VLASS{VLASS_VERSION}%'),
               entities.CosmicDB_Observation.start > cutoff_date,
               entities.CosmicDB_ObservationCalibration.overall_grade <= 0.6)
        .group_by(observation_id_expr)
    )

    with cosmicdb_engine.session() as session:
        results = session.execute(query).fetchall()

    # Extract results
    calibration_counts = [result[1] for result in results]

    # Plot the histogram for the current VLASS_VERSION
    axes[idx].hist(calibration_counts, bins=range(1,max(calibration_counts) + 2), edgecolor='black', align='left')
    axes[idx].set_ylabel(f'VLASS {VLASS_VERSION}\nNumber of Datasets')
    if idx == len(VLASS_VERSIONS) - 1:
        axes[idx].set_xlabel('Calibration Scans Until Grade > 0.6')

# Save the plot
plt.savefig('calibration_scans_until_grade_above_0_6.png', dpi=300)