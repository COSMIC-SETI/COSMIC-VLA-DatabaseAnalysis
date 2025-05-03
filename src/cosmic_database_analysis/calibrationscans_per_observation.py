from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy

# Database setup
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)

VLASS_VERSION = '3.2'  # Adjust this if needed

with cosmicdb_engine.session() as session:
    # Query to get all calibration observations (following original logic)
    calibration_query = (
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

    calibration_results = session.execute(calibration_query).fetchall()

    # Extract dataset IDs from calibration observations
    calibration_scan_ids = set(scan_id for _, scan_id in calibration_results)

    # Count total calibration observations (adhering to your script's logic)
    total_calibration_scans = len(calibration_results)

    # Query to get all observations matching the VLASS version
    all_scans_query = (
        sqlalchemy.select(
            entities.CosmicDB_Observation.scan_id
        )
        .where(entities.CosmicDB_Observation.scan_id.like(f'%VLASS{VLASS_VERSION}%'))
    )

    all_scan_results = session.execute(all_scans_query).fetchall()

    # Count total scans
    total_scans = len(all_scan_results)

    # Count non-calibration scans
    total_non_calibration_scans = total_scans - total_calibration_scans

    # Calculate percentage of calibration scans
    calibration_percentage = (total_calibration_scans / total_scans) * 100 if total_scans > 0 else 0

print(f"Total scans: {total_scans}")
print(f"Total calibration observations: {total_calibration_scans}")
print(f"Total non-calibration scans: {total_non_calibration_scans}")
print(f"Percentage of calibration scans: {calibration_percentage:.2f}%")
