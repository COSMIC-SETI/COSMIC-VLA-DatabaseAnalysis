from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine
import sqlalchemy
import pandas
import multiprocessing as mp

where_clause = entities.CosmicDB_ObservationBeam.id < 1000000000
engine = CosmicDB_Engine(engine_conf_yaml_filepath="/home/cosmic/conf/cosmicdb_conf.yaml")


# mp.set_start_method("fork")
# pool = mp.Pool(processes=12)

def get_first_beam_at_coord(coord):
    res = pandas.read_sql_query(
        sql = sqlalchemy.select(
                entities.CosmicDB_ObservationBeam,
            ).where(
                entities.CosmicDB_ObservationBeam.ra_radians == coord.ra_radians,
                entities.CosmicDB_ObservationBeam.dec_radians == coord.dec_radians,
                where_clause
            ).limit(1),
        con = engine.engine
    )

    # nasty filter for gaia sources which are all numeric
    source = res.source[0]
    try: 
        int(source)
        return res
    except:
        # also accept these
        if source.lower() in ["phase_center", "incoherent"]:
            return res
    
    return None

dfs = []
with mp.Pool(processes=12) as pool:    
    distinct_coords_stmnt = sqlalchemy.select(
        entities.CosmicDB_ObservationBeam.ra_radians,
        entities.CosmicDB_ObservationBeam.dec_radians
    ).where(
        where_clause
    ).distinct()

    with engine.session() as session:
        distinct_coords = [c for c in session.execute(distinct_coords_stmnt)]
    print(f"{len(distinct_coords)} unique co-ordinates...")

    dfs = [r
        for r in pool.map(get_first_beam_at_coord, distinct_coords)
        if r is not None
    ]

df = pandas.concat(dfs, ignore_index=True)
print(df)
df.to_pickle("./beams_of_interest_by_unique_coord.pkl")