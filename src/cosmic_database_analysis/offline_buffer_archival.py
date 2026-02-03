import os, shutil
from multiprocessing import Pool
from cosmic_database import entities
from cosmic_database.engine import CosmicDB_Engine, cli_add_engine_arguments

from cosmic_database_analysis.util_files import find_path 
from cosmic_database_analysis.util_files import is_scan_id, FilepathRegexMatch


import sqlalchemy

import argparse

parser = argparse.ArgumentParser(
    description="Clean up the local /mnt/buf? drives by deleting what cannot be archived, and retroactively archiving what should and can."
)
parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    default=0,
    help="Increase the verbosity of the entity strings."
)
cli_add_engine_arguments(parser)
args = parser.parse_args()
engine_configuration = args.engine_configuration

finding_count = 0

tmp_size = 0
deletion_size = 0

uuids_that_are_full = ["c13ec7fb-58e0-4ab4-a4e4-babd28cc2763", "526ce1bc-e022-488a-a016-bba9adc669ce"]

uuid_map_engineurl = {
    "c13ec7fb-58e0-4ab4-a4e4-babd28cc2763": "mysql+pymysql://cosmic:***@cosmic-storage-3:3307/cosmicobs_storage", # cosmic-storage-3:/srv/cosmicfs13
    "526ce1bc-e022-488a-a016-bba9adc669ce": "mysql+pymysql://cosmic:***@cosmic-storage-3:3308/cosmicobs_storage", # cosmic-storage-3:/srv/cosmicfs14
    "a7b192ae-1127-427d-aa03-b24da8dac6f9": "mysql+pymysql://cosmic:***@cosmic-storage-3:3309/cosmicobs_storage", # cosmic-storage-3:/srv/cosmicfs15
}
uuid_map_archival_size = {uuid: 0 for uuid in uuid_map_engineurl.keys()}
uuid_map_deletion_size = {uuid: 0 for uuid in uuid_map_engineurl.keys()}
uuid_map_networkmount = {}

with CosmicDB_Engine(engine_conf_yaml_filepath=args.engine_configuration, scope=entities.DatabaseScope.Operation).session() as session:
    for uuid in uuid_map_engineurl.keys():
        filesystem_mount = session.scalars(
            sqlalchemy.select(entities.CosmicDB_Filesystem)
            .where(entities.CosmicDB_Filesystem.uuid == uuid)
        ).one().get_latest_mount(session)
        uuid_map_networkmount[
            uuid
        ] = filesystem_mount.network_uri if filesystem_mount.is_current() else None


def cb(findings):
    global engine_configuration, finding_count, filepaths_to_skip, deletion_size, uuid_map_archival_size, uuid_map_deletion_size
    with CosmicDB_Engine(engine_conf_yaml_filepath=args.engine_configuration, scope=entities.DatabaseScope.Operation).session() as session:
        for f_uri in findings:
            # if f_uri in filepaths_to_skip:
            #     continue
            f_size = os.path.getsize(f_uri)

            try:
                filepath = FilepathRegexMatch.from_filepath(f_uri)
            except:
                deletion_size += f_size
                os.remove(f_uri)
                continue
                
            assert is_scan_id(filepath.prefix), f"No discernable scan ID: {filepath}"
            scan_id = filepath.prefix
            
            if f_uri.endswith(".tmp"):
                deletion_size += f_size
                tmp_size += f_size
                os.remove(f_uri)
                continue

            obs_entities = session.scalars(
                sqlalchemy.select(entities.CosmicDB_Observation)
                .where(entities.CosmicDB_Observation.scan_id == scan_id)
            ).all()
            
            if obs_entities is None or len(obs_entities) == 0:
                deletion_size += f_size
                os.remove(f_uri)
                continue

            filesystems = set(oe.archival_filesystem for oe in obs_entities)
            assert len(filesystems) == 1
            filesystem = filesystems.pop()

            with CosmicDB_Engine(
                engine_url=uuid_map_engineurl[filesystem.uuid],
                scope=entities.DatabaseScope.Storage
            ).session() as storsess:
                db_file = storsess.scalars(
                    sqlalchemy.select(entities.CosmicDB_File)
                    .where(entities.CosmicDB_File.local_uri.like(
                        f"%{os.path.basename(f_uri)}"
                    ))
                ).one_or_none()

                if db_file is None or uuid_map_networkmount[filesystem.uuid] is None:
                    if args.verbosity > 0:
                        print(f"Removing: {f_uri}")
                    deletion_size += f_size
                    os.remove(f_uri)
                elif filesystem.uuid in uuids_that_are_full:
                    if args.verbosity > 0:
                        print(f"Deleting: {db_file}")
                    uuid_map_deletion_size[filesystem.uuid] += os.path.getsize(f_uri)
                    deletion_size += f_size
                    os.remove(f_uri)

                    entity_class = entities.CosmicDB_ObservationStamp if f_uri.endswith(".stamps") else entities.CosmicDB_ObservationHit
                    storsess.execute(
                        sqlalchemy.delete(
                            entity_class
                        ).where(
                            entity_class.file_id == db_file.id
                        )
                    )
                else:
                    if args.verbosity > 0:
                        print(f"Archiving: {db_file}")
                    uuid_map_archival_size[filesystem.uuid] += f_size
                    dst_uri = os.path.join(
                        uuid_map_networkmount[filesystem.uuid],
                        db_file.local_uri[1:]
                    )
                    shutil.move(
                        f_uri,
                        dst_uri
                    )

        finding_count += len(findings)

def human_readable_bytes_str(bytesize: int):
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytesize < 1024.0:
            return f"{bytesize:4.3f} {unit}"
        bytesize /= 1024.0

def _error_cb(findings, err):
    raise

with Pool(8) as pool:
    find_path(
        r".*\.(stamps|hits)",
        ["/mnt/buf0/", "/mnt/buf1"],
        pool,
        cb,
        is_file_not_dir=True,
        stop_earliest=True,
        callback_error=_error_cb
    )
    print("total:", finding_count)
    
    for uuid, archival_size in uuid_map_archival_size.items():
        print(uuid_map_networkmount[uuid])
        print("\tArchivals:", human_readable_bytes_str(archival_size))
        print("\tDeletions:", human_readable_bytes_str(uuid_map_deletion_size[uuid]))

    print()
    print(".tmp file deletions:", human_readable_bytes_str(deletion_size))
    print("Deleted:", human_readable_bytes_str(deletion_size))
