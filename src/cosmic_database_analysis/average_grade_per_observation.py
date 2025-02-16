from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy
from sqlalchemy import func  # Import func from sqlalchemy

"""
BASIC REQUIRED STUFF TO GET DATABASE ACCESS
"""
cosmicdb_engine_url = CosmicDB_Engine._create_url("/home/cosmic/conf/cosmicdb_conf.yaml")
cosmicdb_engine = CosmicDB_Engine(engine_url=cosmicdb_engine_url)

# Create a MetaData instance
metadata = sqlalchemy.MetaData()

# Reflect the tables
metadata.reflect(bind=cosmicdb_engine.engine)

"""
QUERY TO FIND AVERAGE GRADE OF OBSERVATION BARRING THE GRADE OF THE FIRST SCAN
"""
# Extract observation ID from scan_id
observation_id_expr = func.substr(entities.CosmicDB_Observation.scan_id, 1, func.length(entities.CosmicDB_Observation.scan_id) - 6)

# Extract scan number from scan_id
scan_number_expr = func.substr(entities.CosmicDB_Observation.scan_id, -5, 3)

# Subquery to find the first scan in each observation with scan_id like '%VLASS3.1%'
first_scan_subquery = (
    sqlalchemy.select(
        observation_id_expr.label('observation_id'),
        func.min(scan_number_expr).label('first_scan_number')
    )
    .where(entities.CosmicDB_Observation.scan_id.like('%VLASS3.1%'))
    .group_by(observation_id_expr)
    .subquery()
)

# Query to calculate the average overall_grade per observation, excluding the first scan
query = (
    sqlalchemy.select(
        observation_id_expr.label('observation_id'),
        func.avg(entities.CosmicDB_ObservationCalibration.overall_grade).label('average_overall_grade')
    )
    .join(
        entities.CosmicDB_Observation,
        entities.CosmicDB_ObservationCalibration.observation_id == entities.CosmicDB_Observation.id
    )
    .join(
        first_scan_subquery,
        (observation_id_expr == first_scan_subquery.c.observation_id) &
        (scan_number_expr != first_scan_subquery.c.first_scan_number)
    )
    .where(entities.CosmicDB_Observation.scan_id.like('%VLASS3.1%'))
    .group_by(observation_id_expr)
)

# Print the query for debugging
print(query)

with cosmicdb_engine.session() as session:
    results = session.execute(query).fetchall()

# Print out the results
for result in results:
    observation_id, average_overall_grade = result
    print(f"Observation ID: {observation_id}, Average Overall Grade: {average_overall_grade}")