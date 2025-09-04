from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import os
import time

import numpy

from blri.times import unix_from_julian_date
from cosmic_database.engine import CosmicDB_Engine, cli_add_engine_arguments, sqlalchemy
from cosmic_database import entities
from seticore.viewer import stamp_capnp, hit_capnp, read_stamps, read_hits, signal_mask, Stamp

# VVV @https://github.com/NMStiegler VVV
# Define a function which finds out how much of the signal is present in each
# antenna. Returns an array of length n_antennas containing floats [0, 1]
# Note: I think this calculates the portion of the INCOHERENT sum of the antennas
def signal_distribution(stamp, hit=None):
    # Get the intensities in the frequency bins of each antenna by summing 
    # over squares of polarization and complex magnitude
    # Also rearrange so indices are (antenna, time bin, frequency bin)
    intensities = numpy.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    
    # Figure out how much signal comes from each antenna
    if hit is None:
        signalmask = stamp.signal_mask()
        signals = [(intensity * signalmask).sum() for intensity in intensities]
    else:
        signalmask = signal_mask( # mask the hit's signal onto the stamp's data (`intensities`)
            hit.signal,
            stamp.stamp.startChannel,
            stamp.stamp.numChannels,
            stamp.stamp.numTimesteps
        )
        hit_offset = hit.signal.index - stamp.stamp.startChannel
        # if hit_offset < 0:
        #     # Not sure when this occurs, so don't handle until the case arises
        #     assert hit.filterbank.startChannel <= stamp.stamp.startChannel

        #     hit_offset += intensities.shape[2]
        #     hit_offset -= hit.filterbank.numChannels
        
        assert hit_offset >= 0, f"Offset {hit_offset} < 0 (stamp.start {stamp.stamp.startChannel}, hit.start {hit.filterbank.startChannel}, reversed {hit.filterbank.foff < 0})"

        signals = [(intensity[:, hit_offset:hit_offset+hit.filterbank.numChannels] * signalmask).sum() for intensity in intensities]
    
    # Return fraction of total signal contributed by each antenna
    total = sum(signals)
    return [signal / total for signal in signals]

# Returns the antenna-index if the stamp is classified as SARFI (otherwise None)
# according to the definition:
# A single antenna contributed >50% of the incoherent signal which caused the 
# hit to be detected as a technosignature in the first place
def is_SARFI(stamp, hit=None):
        sig_dist = signal_distribution(stamp, hit)
        index_max = max(range(len(sig_dist)), key=sig_dist.__getitem__)
        if sig_dist[index_max] > 0.5:
            return  index_max
        return None

def is_hit_in_stamp_universal(
    criteria_ret=None,
    **kwargs
):
    # match coarse channels
    stamp_coarse_channel = kwargs["stamp_coarse_channel"]
    stamp_start_channel = kwargs["stamp_start_channel"]
    stamp_num_channels = kwargs["stamp_num_channels"]
    stamp_tstart_unix = kwargs["stamp_tstart_unix"]
    stamp_num_timesteps = kwargs["stamp_num_timesteps"]
    stamp_tsamp = kwargs["stamp_tsamp"]
    stamp_foff_mhz = kwargs["stamp_foff_mhz"]

    hit_start_channel = kwargs["hit_start_channel"]
    hit_num_channels = kwargs["hit_num_channels"]
    hit_tstart_mjd = kwargs["hit_tstart_mjd"]
    hit_coarse_channel = kwargs["hit_coarse_channel"]
    hit_foff_mhz = kwargs["hit_foff_mhz"]
    hit_num_timesteps = kwargs["hit_num_timesteps"]
    hit_tsamp = kwargs["hit_tsamp"]

    stamp_frequency_channel_range = [
        stamp_start_channel,
        stamp_start_channel+stamp_num_channels
    ]
    hit_frequency_channel_range = [
        hit_start_channel,
        hit_start_channel+hit_num_channels
    ]

    stamp_unix_time_range = [
        stamp_tstart_unix,
        stamp_tstart_unix + stamp_num_timesteps*stamp_tsamp,
    ]
    stamp_tstart_mjd = (stamp_tstart_unix/86400) + 2440587.5 - 2400000.5
    stamp_mjd_time_range = [
        stamp_tstart_mjd,
        stamp_tstart_mjd + (stamp_num_timesteps*stamp_tsamp)/86400,
    ]

    # hits_filterbank_unix_tstart = unix_from_julian_date(hit_tstart_mjd+2400000.5)
    hits_filterbank_unix_tstart = ((hit_tstart_mjd+2400000.5) - 2440587.5)*86400
    hit_unix_time_range = [
        hits_filterbank_unix_tstart,
        hits_filterbank_unix_tstart + hit_num_timesteps*hit_tsamp
    ]
    hit_mjd_time_range = [
        hit_tstart_mjd,
        hit_tstart_mjd + (hit_num_timesteps*hit_tsamp)/86400
    ]
    
    if criteria_ret is None:
        criteria_ret = {}

    # cannot use beam related field as the stamp is antenna data, not beam data
    # criteria_ret[f"RA({stamp_ra} vs {hit_ra})"] = stamp_ra == hit_ra
    # criteria_ret[f"DEC({stamp_dec} vs {hit_dec})"] = stamp_dec == hit_dec
    # criteria_ret[f"BEAM({stamp_signal.beam} vs {hit.signal.beam})"] = stamp_signal.beam == hit.signal.beam

    criteria_ret[f"CoarseChannel"] = stamp_coarse_channel == hit_coarse_channel
    criteria_ret[f"Frequency Resolution({stamp_foff_mhz} isclose {hit_foff_mhz})"] = stamp_foff_mhz == hit_foff_mhz # numpy.isclose(stamp_foff_mhz, hit_foff_mhz)
    criteria_ret[f"TStartUnix_lower({stamp_unix_time_range[0]} S<=H {hit_unix_time_range[0]})"] = stamp_unix_time_range[0] <= hit_unix_time_range[0]
    criteria_ret[f"TStartUnix_upper({stamp_unix_time_range[1]} S>=H {hit_unix_time_range[1]})"] = stamp_unix_time_range[1] >= hit_unix_time_range[1]
    criteria_ret[f"TStartMjd_lower({stamp_mjd_time_range[0]} S<=H {hit_mjd_time_range[0]})"] = stamp_mjd_time_range[0] <= hit_mjd_time_range[0]
    criteria_ret[f"TStartMjd_upper({stamp_mjd_time_range[1]} S>=H {hit_mjd_time_range[1]})"] = stamp_mjd_time_range[1] >= hit_mjd_time_range[1]
    criteria_ret[f"FrequencyRange_lower({stamp_frequency_channel_range[0]} S<=H {hit_frequency_channel_range[0]})"] = stamp_frequency_channel_range[0] <= hit_frequency_channel_range[0]
    criteria_ret[f"FrequencyRange_upper({stamp_frequency_channel_range[1]} S>=H {hit_frequency_channel_range[1]})"] = stamp_frequency_channel_range[1] >= hit_frequency_channel_range[1]
    return all(criteria_ret.values())
# ^^^ @https://github.com/NMStiegler ^^^

def is_hit_in_stamp(hit, stamp, criteria_ret=None):
    return is_hit_in_stamp_universal(
        criteria_ret,

        stamp_coarse_channel = stamp.stamp.coarseChannel,
        stamp_start_channel = stamp.stamp.startChannel,
        stamp_num_channels = stamp.stamp.numChannels,
        stamp_foff_mhz = stamp.stamp.foff,
        stamp_tstart_unix = stamp.stamp.tstart,
        stamp_num_timesteps = stamp.stamp.numTimesteps,
        stamp_tsamp = stamp.stamp.tsamp,

        hit_coarse_channel = hit.filterbank.coarseChannel,
        hit_start_channel = hit.filterbank.startChannel,
        hit_num_channels = hit.filterbank.numChannels,
        hit_foff_mhz = hit.filterbank.foff,
        hit_tstart_mjd = hit.filterbank.tstart,
        hit_num_timesteps = hit.filterbank.numTimesteps,
        hit_tsamp = hit.filterbank.tsamp,
    )

def is_hit_in_stamp_dbentity(hit, stamp, criteria_ret=None):
    return is_hit_in_stamp_universal(
        criteria_ret,

        stamp_coarse_channel = stamp.coarse_channel,
        stamp_start_channel = stamp.start_channel,
        stamp_num_channels = stamp.num_channels,
        stamp_foff_mhz = stamp.foff_mhz,
        stamp_tstart_unix = stamp.tstart,
        stamp_num_timesteps = stamp.num_timesteps,
        stamp_tsamp = stamp.tsamp,

        hit_coarse_channel = hit.coarse_channel,
        hit_start_channel = hit.start_channel,
        hit_num_channels = hit.num_channels,
        hit_foff_mhz = hit.foff_mhz,
        hit_tstart_mjd = hit.tstart,
        hit_num_timesteps = hit.num_timesteps,
        hit_tsamp = hit.tsamp,
    )


@dataclass
class HitRelations:
    largest_stamp: int
    subsumed_stamps: List[int]
    overlapping_stamps: List[int]

    def __str__(self) -> str:
        s = f"HitRelations({self.largest_stamp}"
        if len(self.subsumed_stamps) > 0:
            s += f", subsumed = {self.subsumed_stamps}"
        if len(self.overlapping_stamps) > 0:
            s += f", overlapping = {self.subsumed_stamps}"
        return s+")"

@dataclass
class SARFI_SeiveResult:
    hit_index_SARFI_map: Dict[int, int]
    stamp_index_map_hit_indices: Dict[int, List[int]]
    hit_relations: List[Optional[HitRelations]]

def stringify(seticore_capnp, type_str=None, indent_level=0):
    if type_str is None:
        assert False, "this doesn't work"
        if isinstance(seticore_capnp, Stamp):
            return stringify(seticore_capnp.stamp, "stamp")
        elif isinstance(seticore_capnp, stamp_capnp.Stamp):
            type_str = "stamp"
        elif isinstance(seticore_capnp, hit_capnp.Hit):
            type_str = "hit"
        elif isinstance(seticore_capnp, hit_capnp.Signal):
            type_str = "signal"
        elif isinstance(seticore_capnp, hit_capnp.Filterbank):
            type_str = "filterbank"
        else:
            assert False, f"Cannot handle {type(seticore_capnp)}"

    indent_level += 1
    indent="    "*indent_level
    if type_str == "viewer_stamp":
        return stringify(seticore_capnp.stamp, type_str="stamp", indent_level=indent_level)
    elif type_str == "stamp":
        return f"""(
        {indent}sourceName = {seticore_capnp.sourceName},
        {indent}ra = {seticore_capnp.ra},
        {indent}dec = {seticore_capnp.dec},
        {indent}fch1 = {seticore_capnp.fch1},
        {indent}foff = {seticore_capnp.foff},
        {indent}ftop = {seticore_capnp.fch1+seticore_capnp.foff*seticore_capnp.numChannels},
        {indent}tstart = {seticore_capnp.tstart},
        {indent}tsamp = {seticore_capnp.tsamp},
        {indent}tstop = {seticore_capnp.tstart+seticore_capnp.tsamp*seticore_capnp.numTimesteps/86400},
        {indent}telescopeId = {seticore_capnp.telescopeId},
        {indent}numTimesteps = {seticore_capnp.numTimesteps},
        {indent}numChannels = {seticore_capnp.numChannels},
        {indent}numPolarizations = {seticore_capnp.numPolarizations},
        {indent}numAntennas = {seticore_capnp.numAntennas},
        {indent}data = [...],
        {indent}coarseChannel = {seticore_capnp.coarseChannel},
        {indent}fftSize = {seticore_capnp.fftSize},
        {indent}startChannel = {seticore_capnp.startChannel},
        {indent}signal = {seticore_capnp.signal},
        {indent}schan = {seticore_capnp.schan},
        {indent}obsid = {seticore_capnp.obsid},
        )"""
    elif type_str == "hit":
        return f"""(
        {indent}signal = {stringify(seticore_capnp.signal, type_str='signal', indent_level=indent_level)},
        {indent}filterbank = {stringify(seticore_capnp.filterbank, type_str='filterbank', indent_level=indent_level)}\n)"""
    elif type_str == "signal":
        return str(seticore_capnp)
    elif type_str == "filterbank":
        tstart = unix_from_julian_date(seticore_capnp.tstart+2400000.5)
        return f"""(
        {indent}sourceName = "{seticore_capnp.sourceName}",
        {indent}fch1 = {seticore_capnp.fch1},
        {indent}foff = {seticore_capnp.foff},
        {indent}ftop = {seticore_capnp.fch1+seticore_capnp.foff*seticore_capnp.numChannels},
        {indent}tstart (JD) = {seticore_capnp.tstart},
        {indent}tsamp = {seticore_capnp.tsamp},
        {indent}trange (unix) = [{tstart},{tstart + seticore_capnp.numTimesteps*seticore_capnp.tsamp/86400}],
        {indent}ra = {seticore_capnp.ra},
        {indent}dec = {seticore_capnp.dec},
        {indent}telescopeId = {seticore_capnp.telescopeId},
        {indent}numTimesteps = {seticore_capnp.numTimesteps},
        {indent}numChannels = {seticore_capnp.numChannels},
        {indent}coarseChannel = {seticore_capnp.coarseChannel},
        {indent}startChannel = {seticore_capnp.startChannel},
        {indent}data = [...]
        )"""

def is_stamp_smaller_universal(
    stampA_start_channel,
    stampA_num_channels,
    stampA_tstart,
    stampA_tsamp,
    stampA_num_timesteps,
    stampB_start_channel,
    stampB_num_channels,
    stampB_tstart,
    stampB_tsamp,
    stampB_num_timesteps,
) -> bool:
    # returns stampA < stampB, ie that stampB spans stampA completely and more
    # assume equivalent tsamp, foff, coarse_channel...

    stampA_ch_first = stampA_start_channel
    stampB_ch_first = stampB_start_channel
    stampA_ch_last = stampA_start_channel+stampA_num_channels
    stampB_ch_last = stampB_start_channel+stampB_num_channels

    stampA_tstop = stampA_tstart+stampA_num_timesteps*stampA_tsamp
    stampB_tstop = stampB_tstart+stampB_num_timesteps*stampB_tsamp
    
    if stampB_ch_last < stampA_ch_last:
        return False
    if stampB_ch_first > stampA_ch_first:
        return False
    
    if stampB_tstop < stampA_tstop:
        return False
    if stampB_tstart > stampA_tstart:
        return False

    return True

def is_stamp_smaller(stampA, stampB) -> bool:
    return is_stamp_smaller_universal(
        stampA.stamp.startChannel,
        stampA.stamp.numChannels,
        stampA.stamp.tstart,
        stampA.stamp.numTimesteps,
        stampA.stamp.tsamp,
        stampB.stamp.startChannel,
        stampB.stamp.numChannels,
        stampB.stamp.tstart,
        stampB.stamp.numTimesteps,
        stampB.stamp.tsamp,
    )

def is_stamp_smaller_dbentity(stampA, stampB) -> bool:
    return is_stamp_smaller_universal(
        stampA.start_channel,
        stampA.num_channels,
        stampA.tstart,
        stampA.num_timesteps,
        stampA.tsamp,
        stampB.start_channel,
        stampB.num_channels,
        stampB.tstart,
        stampB.num_timesteps,
        stampB.tsamp,
    )

def match_hits_to_stamps(
    stamps: Union[List[stamp_capnp.Stamp], List[entities.CosmicDB_ObservationStamp]],
    hits: Union[List[hit_capnp.Hit], List[entities.CosmicDB_ObservationHit]],
    match_first: bool = True,
    assertions: bool = True,
    db_entities: bool = False
) -> Tuple[
    Dict[int, List[int]],
    List[Optional[HitRelations]],
]:
    stamp_index_map_hit_indices = {}
    hit_relations = []
    
    for hit_i, hit in enumerate(hits):
        # stamp_indices_found = [] # ordered with largest stamp span first
        relations = None
        criteria_list = []
        for stamp_i, stamp in enumerate(stamps):
            if stamp_i not in stamp_index_map_hit_indices:
                stamp_index_map_hit_indices[stamp_i] = []
            
            criteria = {}
            if not db_entities:
                found = is_hit_in_stamp(hit, stamp, criteria_ret=criteria)
            else:
                found = is_hit_in_stamp_dbentity(hit, stamp, criteria_ret=criteria)
            criteria_list.append(criteria)
            if not found:
                continue

            if relations is None:
                relations = HitRelations(
                    stamp_i,
                    [],
                    []
                )
                stamp_index_map_hit_indices[stamp_i].append(hit_i)
                if match_first:
                    break
                continue

            # if this stamp is bigger than the last,
            # swap hit affiliation
            
            # there must be some overlap if the hit is seen in more than one
            stampA = stamps[relations.largest_stamp]
            stampB = stamps[stamp_i]

            if not db_entities:
                a_smaller_than_b = is_stamp_smaller(stampB, stampA)
                b_smaller_than_a = is_stamp_smaller(stampA, stampB)
            else:
                a_smaller_than_b = is_stamp_smaller_dbentity(stampB, stampA)
                b_smaller_than_a = is_stamp_smaller_dbentity(stampA, stampB)
            if a_smaller_than_b:
                # previously tagged stamp is larger and fully overlapping
                relations.subsumed_stamps.append(stamp_i)
                continue
            if b_smaller_than_a:
                assert hit_i == stamp_index_map_hit_indices[relations.largest_stamp].pop()
                relations.subsumed_stamps.append(relations.largest_stamp)

                relations.largest_stamp =  stamp_i
                stamp_index_map_hit_indices[stamp_i].append(hit_i)
                continue
            
            if False and assertions:
                if not db_entities:
                    stampA_str = stringify(stampA.stamp, type_str='stamp')
                    stampB_str = stringify(stampB.stamp, type_str='stamp')
                else:
                    stampA_str = str(stampA)
                    stampB_str = str(stampB)
                raise AssertionError(f"Stamps cover different, probably overlapping, data: \n#{relations.largest_stamp}:\n\t{stampA_str}\n\n#{stamp_i}:\n\t{stampB_str}")
            relations.overlapping_stamps.append(stamp_i)

        if relations is None and assertions:
            if not db_entities:
                hit_str = stringify(hit, type_str='hit')
            else:
                hit_str = str(hit)
            raise AssertionError(f"No stamp found for the hit (#{hit_i}): \n{hit_str}\n"+"\n".join(f"#{i}:\t{c}" for i, c in enumerate(criteria_list) if c['CoarseChannel']))
        hit_relations.append(relations)

    return stamp_index_map_hit_indices, hit_relations

def seive_SARFI_stamp(
    stamps: List[stamp_capnp.Stamp],
    hits: List[hit_capnp.Hit],
    sanity_check_tophit: bool = True,
    sanity_check_is_SARFI: bool = False,
    assertions: bool = False,
) -> SARFI_SeiveResult:

    stamp_index_map_hit_indices, hit_relations = match_hits_to_stamps(stamps, hits, assertions=assertions)

    hit_index_SARFI_map = {}
    for stamp_i, stamp in enumerate(stamps):
        if len(stamp_index_map_hit_indices[stamp_i]) == 0:
            continue
    
        for i, hit_i in enumerate(stamp_index_map_hit_indices[stamp_i]):
            sarfi_index = is_SARFI(stamp, hit=hits[hit_i])
            if sarfi_index is not None:
                hit_index_SARFI_map[hit_i] = sarfi_index

        # below is a sanity check...
        # is_SARFI(stamp) === is_SARFI(stamp, hit=stamp.topHit)
        if not (sanity_check_tophit or sanity_check_is_SARFI):
            continue
        
        stamp_signal_dict = stamp.stamp.signal.to_dict()
        top_hit = [
            stamp_signal_dict == hits[hit_i].signal.to_dict()
            for hit_i in stamp_index_map_hit_indices[stamp_i]
        ]
        assert sum(top_hit) == 1, f"No related hit is stamp's TopHit!\n{stamp_signal_dict}\n {[hits[hit_i].signal.to_dict() for hit_i in stamp_index_map_hit_indices[stamp_i]]}"
        if not sanity_check_is_SARFI:
            continue

        top_hit_index = top_hit.index(True)
        top_hit = hits[stamp_index_map_hit_indices[stamp_i][top_hit_index]]

        stamp_details = {"startChannel": stamp.stamp.startChannel, "numChannels": stamp.stamp.numChannels, "numTimesteps": stamp.stamp.numTimesteps}
        tophit_details = {"startChannel": top_hit.filterbank.startChannel, "numChannels": top_hit.filterbank.numChannels, "numTimesteps": top_hit.filterbank.numTimesteps}
        assert hit_index_SARFI_map.get(top_hit_index, None) == is_SARFI(stamp), f"Top hit vs stamp SARFI mismatch: stamp({stamp_details}), tophit({tophit_details})"

    return SARFI_SeiveResult(
        hit_index_SARFI_map,
        stamp_index_map_hit_indices,
        hit_relations
    )


def write_nonSARFI_stamps_and_hits(
    result: SARFI_SeiveResult,
    stamps: str,
    hits: str,
    output_filepath_stamps: str,
    output_filepath_hits: str,
    validate_output = False
):
    output_filepath_stamps is None

    valid_hit_indices = []
    with open(output_filepath_stamps, 'w') as fio:
        for stamp_i, hit_indices in enumerate(result.stamp_index_map_hit_indices):
            if all(
                result.hit_index_SARFI_map.get(h_i) is not None
                for h_i in hit_indices
            ):
                continue
            stamp_wr = stamp_capnp.Stamp.new_message()
            stamp_wr.from_dict(stamps[stamp_i].stamp.to_dict())
            stamp_wr.write(fio)

            valid_hit_indices.extend(
                result.stamp_index_map_hit_indices[stamp_i]
            )

    valid_hit_indices.sort()
    with open(output_filepath_hits, 'w') as fio:
        for hit_i in valid_hit_indices:
            hit_wr = hit_capnp.Hit.new_message()
            hit_wr.from_dict(hits[hit_i].to_dict())
            hit_wr.write(fio)

    if validate_output:
        valid_stamp_indices = [
            s_i 
            for s_i, hit_indices in result.stamp_index_map_hit_indices.items()
            if all(
                result.hit_index_SARFI_map.get(h_i) is not None
                for h_i in hit_indices
            ) == False
        ]
        for i, wrapper in enumerate(read_stamps(output_filepath_stamps)):
            assert(wrapper.stamp.to_dict() == stamps[valid_stamp_indices[i]].stamp.to_dict())

        for i, hit in enumerate(read_hits(output_filepath_hits)):
            assert(hit.to_dict() == hits[valid_hit_indices[i]].to_dict())

    return result


def _insort(cont:list, elem, key=lambda x: x):
    index = 0
    while index < len(cont) and key(cont[index]) <= key(elem):
        index += 1
    cont.insert(index, elem)


def _filter_filepaths_to_process_in_subband(
    related_filepath,
    observation_id,
    subband_offset,
    tuning,
    session
) -> Dict[str, List[str]]:
    hit_file_uri_map_enum_id_map: Dict[str, Dict[int, int]] = {}
    stamp_file_uri_map_enum_id_map: Dict[str, Dict[int, int]] = {}

    for h in session.scalars(
        sqlalchemy.select(entities.CosmicDB_ObservationHit)
        .where(entities.CosmicDB_ObservationHit.observation_id == observation_id)
        .where(entities.CosmicDB_ObservationHit.subband_offset == subband_offset)
        .where(entities.CosmicDB_ObservationHit.tuning == tuning)
        .order_by(entities.CosmicDB_ObservationHit.file_local_enumeration.asc())
    ).all():
        assert h.file is not None, f"{related_filepath}: {h}"
        if h.file.uri not in hit_file_uri_map_enum_id_map:
            hit_file_uri_map_enum_id_map[h.file.uri] = {}
        hit_file_uri_map_enum_id_map[h.file.uri][h.file_local_enumeration] = h.id
    
    # print(hit_file_uri_map_enum_id_map)
    for s in session.scalars(
        sqlalchemy.select(entities.CosmicDB_ObservationStamp)
        .where(entities.CosmicDB_ObservationStamp.observation_id == observation_id)
        .where(entities.CosmicDB_ObservationStamp.subband_offset == subband_offset)
        .where(entities.CosmicDB_ObservationStamp.tuning == tuning)
        .order_by(entities.CosmicDB_ObservationStamp.file_local_enumeration.asc())
    ).all():
        assert s.file is not None, f"{related_filepath}: {s}"
        if s.file.uri not in stamp_file_uri_map_enum_id_map:
            stamp_file_uri_map_enum_id_map[s.file.uri] = {}
        stamp_file_uri_map_enum_id_map[s.file.uri][s.file_local_enumeration] = s.id
    # print(stamp_file_uri_map_enum_id_map)

    file_paths = [f for f in hit_file_uri_map_enum_id_map.keys()]
    file_paths.extend([f for f in stamp_file_uri_map_enum_id_map.keys()])
    stem_map_suffixes: Dict[str, List[str]] = {}
    for f in file_paths:
        stem, suffix = f.split(".seticore.", 1)
        if stem not in stem_map_suffixes:
            stem_map_suffixes[stem] = []
        stem_map_suffixes[stem].append(suffix)
    # print(f"\t{stem_map_suffixes}")

    stems = [k for k in stem_map_suffixes.keys()]
    for stem in stems:
        all_processed = True
        for suffix in stem_map_suffixes[stem]:
            uri = f"{stem}.seticore.{suffix}"
            item_flagged = None
            if suffix.endswith("hits"):
                local_enum, first_hit_id = next(iter(hit_file_uri_map_enum_id_map[uri].items()))
                item_flagged = session.scalars(
                    sqlalchemy.select(entities.CosmicDB_HitFlagSARFI)
                    .where(entities.CosmicDB_HitFlags.hit_id == first_hit_id)
                ).first()
            else:
                assert suffix.endswith("stamps"), f"Not expecting {uri}"
                local_enum, first_stamp_id = next(iter(stamp_file_uri_map_enum_id_map[uri].items()))
                item_flagged = session.scalars(
                    sqlalchemy.select(entities.CosmicDB_ObservationHit)
                    .where(entities.CosmicDB_ObservationHit.stamp_id == first_stamp_id)
                ).first()

            if item_flagged is None:
                all_processed = False
                break
        if not all_processed:
            stem_map_suffixes[stem].sort()
            continue
        
        # all processed, drop stem
        stem_map_suffixes.pop(stem)

    # stem_map_suffixes now contains only those files which need to be filtered
    return stem_map_suffixes

@dataclass
class Timestamper:
    stamps: List[float]
    descriptions: List[str]

    def __init__(self, desc: str):
        self.stamps = [time.time()]
        self.descriptions = [desc]

    def stamp(self, desc: str):
        self.stamps.append(time.time())
        self.descriptions.append(desc)
    
    def __str__(self) -> str:
        last_elapsed = time.time() - self.stamps[-1]
        elapsed = [
            self.stamps[i] - self.stamps[i-1]
            for i in range(1, len(self.stamps))
        ]
        elapsed_sum = sum(elapsed) + last_elapsed
        s = f"Timestamps: {self.descriptions[0]}"
        if len(elapsed) > 1:
            s += "\n" + "\n".join(
                f"\tafter {e*1000:0.3f} ms: {self.descriptions[i+1]} ({100*e/elapsed_sum:0.3f} %)"
                for i, e in enumerate(elapsed)
            )
        s += f"\n\tafter {last_elapsed*1000:0.3f} ms: now ({100*last_elapsed/elapsed_sum:0.3f} %)"
        s += f"\nTotal Elapsed: {elapsed_sum*1000:0.3f} ms"
        return s

def cli_db_tag_sarfi(
    engine: CosmicDB_Engine
):
    with engine.session() as session:
        id_batch_size = 10
        id_lowest = session.scalars(
            sqlalchemy.select(
                entities.CosmicDB_ObservationStamp.observation_id
            )
            .order_by(entities.CosmicDB_ObservationStamp.observation_id.asc())
            .limit(1)
        ).one()
        id_highest = session.scalars(
            sqlalchemy.select(
                entities.CosmicDB_ObservationStamp.observation_id
            )
            .order_by(entities.CosmicDB_ObservationStamp.observation_id.desc())
            .limit(1)
        ).one()
        id_rolling = 44199

        while id_rolling < id_highest:
            ss = session.execute(
                sqlalchemy.select(
                    entities.CosmicDB_ObservationStamp.observation_id,
                    entities.CosmicDB_ObservationStamp.subband_offset,
                    entities.CosmicDB_ObservationStamp.tuning,
                    entities.CosmicDB_ObservationStamp.file_id,
                )
                .order_by(entities.CosmicDB_ObservationStamp.observation_id.asc())
                .where(entities.CosmicDB_ObservationStamp.observation_id >= id_rolling)
                .where(entities.CosmicDB_ObservationStamp.observation_id < id_rolling+id_batch_size)
                # .limit(3)
                .distinct()
            )
            id_rolling += id_batch_size

            for (observation_id, subband_offset, tuning, file_id) in ss:
                print(f"ObsId: {observation_id}, subband: {tuning}-C{subband_offset}")
                if file_id is None:
                    print(f"No file, skipping.")
                    continue
                s_file = session.scalars(
                    sqlalchemy.select(
                        entities.CosmicDB_File,
                    )
                    .where(
                        entities.CosmicDB_File.id == file_id
                    )
                ).one()
                if not os.path.exists(s_file.uri):
                    print(f"Can't reach file, skipping: {s_file.uri}")
                    continue

                ts = Timestamper("SARFI Operation")

                stem_map_suffixes = _filter_filepaths_to_process_in_subband(
                    s_file.uri,
                    observation_id,
                    subband_offset,
                    tuning,
                    session
                )
                print(f"\t{stem_map_suffixes}")
                ts.stamp("filter filepaths")
                print(ts)

                for stem, suffixes in stem_map_suffixes.items():
                    hits = []
                    stamps = []
                    stamps_suffix_tuple_enum_range = []
                    ts = Timestamper("SARFI on files")

                    flags_to_commit = []
                    for suffix in suffixes:
                        filepath = f"{stem}.seticore.{suffix}"
                        if not os.path.exists(filepath):
                            file_entity = session.scalars(
                                sqlalchemy.select(
                                    entities.CosmicDB_File
                                )
                                .where(entities.CosmicDB_File.uri == filepath)
                            ).one_or_none()

                            if file_entity is None:
                                file_entity = entities.CosmicDB_File(
                                    uri =filepath
                                )
                                session.add(file_entity)
                                session.commit()


                            flags_to_commit.append(
                                entities.CosmicDB_FileFlags(
                                    file_id=file_entity.id,
                                    missing=True
                                )
                            )
                            continue

                        if suffix.endswith("hits"):
                            assert len(hits) == 0, f"Not expecting more than one hits file: {suffixes}."
                            hits.extend(read_hits(filepath))
                        else:
                            assert suffix.endswith("stamps")
                            first_index = len(stamps)
                            stamps.extend(read_stamps(filepath))
                            last_index = len(stamps)
                            stamps_suffix_tuple_enum_range.append((suffix, range(first_index, last_index)))

                    if len(flags_to_commit) > 0:
                        ts.stamp(f"commit {len(flags_to_commit)} flags")
                        session.add_all(flags_to_commit)
                        session.commit()
                    ts.stamp("filter filepaths")

                    seive_result = seive_SARFI_stamp(stamps, hits)
                    ts.stamp("sieve stamps")
                    # print(seive_result)

                    hit_orm_ids = []
                    # stamp_subsumed_sets = [set() for i in range(len(stamps))]
                    stamp_index_map_consumed_by_index = {}
                    for hit_index, hit_relations in enumerate(seive_result.hit_relations):
                        hit_entity_id = [h.id for h in session.execute(
                            sqlalchemy.select(
                                entities.CosmicDB_ObservationHit.id
                            ).join(
                                entities.CosmicDB_File
                            ).where(
                                entities.CosmicDB_ObservationHit.file.uri == f"{stem}.seticore.hits",
                                entities.CosmicDB_ObservationHit.file_local_enumeration == hit_index,
                            )
                        )]
                        assert len(hit_entity_id) == 1, f"{len(hit_entity_id)} hit entities found with '{stem}'.seticore.hits and local enumeration {hit_index}.\n{hit_entity_id}"
                        hit_entity_id = hit_entity_id[0]
                        hit_orm_ids.append(hit_entity_id)

                        if hit_relations is None:
                            continue
                        
                        for subsumed_stamp_index in hit_relations.subsumed_stamps:
                            if subsumed_stamp_index not in stamp_index_map_consumed_by_index:
                                stamp_index_map_consumed_by_index[subsumed_stamp_index] = hit_relations.largest_stamp
                            else:
                                # shouldn't ever trigger...
                                assert stamp_index_map_consumed_by_index[subsumed_stamp_index] == hit_relations.largest_stamp, f"Need to handle multiple consumptions..."

                        if len(hit_relations.overlapping_stamps) == 0:
                            continue
                        assert False, f"Need to handle overlapping stamps...." # by implementing a flag for overlaps


                    for tup in stamps_suffix_tuple_enum_range:
                        stamps_suffix, enum_range = tup
                        stamp_orm_ids = [s for s in session.execute(
                            sqlalchemy.select(
                                entities.CosmicDB_ObservationStamp.id
                            ).join(
                                entities.CosmicDB_File
                            ).where(
                                entities.CosmicDB_ObservationStamp.file.uri == f"{stem}.seticore.{stamps_suffix}",
                            ).order_by(
                                entities.CosmicDB_ObservationStamp.file_local_enumeration.asc()
                            )
                        )]
                        assert len(enum_range) == len(stamp_orm_ids) # TODO flag this stamps file, as what though..

                        db_updates = 0
                        db_entities = []
                        for local_file_enum, stamps_index in enumerate(enum_range):
                            print(f"{stamps_suffix}@{local_file_enum} -> {stamps_index} ")

                            if stamps_index in stamp_index_map_consumed_by_index:
                                assert False, f"Need to handle overlapping stamps...."
                                redundant_to = stamp_orm_ids[
                                    stamp_index_map_consumed_by_index[stamps_index]
                                ]
                            
                                is_new, stamp_flag_entity = engine.update_entity(
                                    session,
                                    entities.CosmicDB_StampFlagRedundant(
                                        stamp_id=stamp_orm_ids[local_file_enum],
                                        redundant_to=redundant_to
                                    ),
                                    field_update_filter=("sarfi", "no_hits")
                                )
                                if is_new:
                                    db_entities.append(stamp_flag_entity)
                                else:
                                    db_updates += 1

                            for hit_index in seive_result.stamp_index_map_hit_indices[stamps_index]:
                                hit_entity = session.scalars(
                                    sqlalchemy.select(
                                        entities.CosmicDB_ObservationHit.id
                                    ).join(
                                        entities.CosmicDB_File
                                    ).where(
                                        entities.CosmicDB_ObservationHit.file.uri == f"{stem}.seticore.hits",
                                        entities.CosmicDB_ObservationHit.file_local_enumeration == hit_index,
                                    )
                                ).one()
                                hit_entity.stamp_id = stamp_orm_ids[local_file_enum]

                                if hit_index in seive_result.hit_index_SARFI_map:
                                    is_new, hit_flag_entity = engine.update_entity(
                                        session,
                                        entities.CosmicDB_HitFlagSARFI(
                                            hit_id=hit_entity.id,
                                            antenna_index=seive_result.hit_index_SARFI_map[hit_index],
                                        ),
                                        field_update_filter=("antenna_index")
                                    )
                                    if is_new:
                                        db_entities.append(hit_flag_entity)
                                    else:
                                        db_updates += 1


                        
                        print(f"{db_updates} updates, {len(db_entities)} new entities")
                        session.commit()

                    print(ts)


if __name__ == "__main__":


    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retro-actively process hits to tag those that are SARFI."
    )
    cli_add_engine_arguments(parser)
    args = parser.parse_args()

    engine = CosmicDB_Engine(
        engine_conf_yaml_filepath=args.engine_configuration,
        scope=entities.DatabaseScope.Storage
    )
    with engine.session() as session:
        stamps, hits = [
            session.scalars(
                sqlalchemy.select(
                    e
                ).where(
                    e.observation_id == 231,
                    e.tuning == "BD",
                    e.subband_offset == 352,
                )
            ).all()
            for e in [
                entities.CosmicDB_ObservationStamp,
                entities.CosmicDB_ObservationHit,
            ]
        ]
        stamp_index_map_hit_indices, hit_relations = match_hits_to_stamps(stamps, hits, assertions=False, db_entities=True)
        for stamp_i, related_hitindices in stamp_index_map_hit_indices.items():
            if stamps[stamp_i].file_local_enumeration == 3:
                print(f"stamp_index_map_hit_indices[3]: {related_hitindices}")
                print(f"stamp[3]: {stamps[stamp_i]}")
                for hit_i in related_hitindices:
                    print(f"hit_relations[{hit_i}]: {hit_relations[hit_i]}")
                    print(f"hits[{hit_i}]: {hits[hit_i]}")
                break
    
    exit(0)
    # stamps = []
    # stamps.extend(read_stamps("/mnt/cosmic-storage-3/cosmicfs13/voyager_benchmark/TCOS0001_sb49105488_1_1.60901.90559458333/TCOS0001_sb49105488_1_1.60901.90559458333.4.1/TCOS0001_sb49105488_1_1.60901.90559458333.4.1.BD.C352.0000.raw.seticore.0000.stamps"))
    # hits = []
    # hits.extend(  read_hits("/mnt/cosmic-storage-3/cosmicfs13/voyager_benchmark/TCOS0001_sb49105488_1_1.60901.90559458333/TCOS0001_sb49105488_1_1.60901.90559458333.4.1/TCOS0001_sb49105488_1_1.60901.90559458333.4.1.BD.C352.0000.raw.seticore.hits"))

    # stamp_index_map_hit_indices, hit_relations = match_hits_to_stamps(stamps, hits, assertions=False)
    # print(f"stamp_index_map_hit_indices[3]: {stamp_index_map_hit_indices[3]}")
    # print(f"len(hit_relations): {len(hit_relations)}")
    # print(f"hit_relations[9]: {hit_relations[9]}")
    # exit(0)
    
    
    with engine.session() as session:
        stamp = session.scalars(
            sqlalchemy.select(
                entities.CosmicDB_ObservationStamp
            ).where(
                entities.CosmicDB_ObservationStamp.id == 2008909
            )
        ).one()
        print(stamp)
        for hit in session.scalars(
            sqlalchemy.select(
                entities.CosmicDB_ObservationHit
            ).where(
                entities.CosmicDB_ObservationHit.id.in_([
                    4490465,
                    4490473,
                    4490481,

                    4491244,
                    4491252,
                    4491261,
                    4491270,
                    4491279,
                    4491286,
                ])
            )
        ).all():
            crit_d = {}
            contained = is_hit_in_stamp_dbentity(
                hit,
                stamp,
                crit_d
            )
            print(f"\nH{hit.id} in S{stamp.id}: {contained}\n\t{crit_d}")
            if contained:
                print(hit)
                print(stamp.file.local_uri, stamp.file_local_enumeration)
                print(hit.file.local_uri, hit.file_local_enumeration)

    exit()
    # cli_db_tag_sarfi(
    #     engine
    # )