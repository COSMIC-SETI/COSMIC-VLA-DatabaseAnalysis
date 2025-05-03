from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy

# Database setup
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)

VLASS_VERSION = '3.1'  # Adjust this if needed

with cosmicdb_engine.session() as session:
    # Query to get all calibration observations
    query = (
        sqlalchemy.select(
            entities.CosmicDB_Observation.id,
            entities.CosmicDB_Observation.scan_id
        )
        .join(
            entities.CosmicDB_ObservationCalibration,
            entities.CosmicDB_ObservationCalibration.observation_id == entities.CosmicDB_Observation.id
        )
        .where(entities.CosmicDB_Observation.scan_id.like(f'%VLASS{VLASS_VERSION}%'))
    )

    results = session.execute(query).fetchall()

    # Count total calibration observations
    total_calibration_observations = len(results)

    # Extract dataset IDs
    unique_datasets = set('.'.join(scan_id.split('.')[:-2]) for _, scan_id in results)

    # Count total datasets
    total_datasets = len(unique_datasets)

print(f"Total calibration observations: {total_calibration_observations}")
print(f"Total unique datasets: {total_datasets}")
