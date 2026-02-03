from dataclasses import dataclass
import os, re

filepath_regex = r"(?P<dir>.*/)(?P<prefix>.*)\.(?P<tuning>(AC|BD)(\_\dBIT)?)\.C(?P<subband>\d+)\.(?P<suffix>.*)"
dataset_id_regex = r"(.*?\.(5|6)\d{4}\.\d{6,})"
scan_id_regex = r"(.*?\.(5|6)\d{4}\.\d{6,}\.\d+\.\d+)"

@dataclass
class FilepathRegexMatch:
    dir: str
    prefix: str
    tuning: str
    subband: str
    suffix: str

    def from_filepath(fp: str):
        m = re.match(filepath_regex, fp)
        assert m is not None, f"Unrecognised filepath format: {fp}"
        return FilepathRegexMatch(**m.groupdict())

if __name__ == "__main__":
    m = re.match(filepath_regex, "/mnt/cosmic-storage-1/data0/vlass_target/VLASS3.1.sb43319939.eb43644901.59991.14533251157.100.1/VLASS3.1.sb43319939.eb43644901.59991.14533251157.100.1.AC_8BIT.C256.0002.seticore.hits")
    assert m is not None
    FilepathRegexMatch(**m.groupdict())

def extract_mjd(s: str):
    m = re.findall(r"(5|6\d{4})\.(\d{6,})", s)
    if len(m) == 0:
        raise RuntimeError(f"No MJD found in '{s}'")
    assert len(m) == 1, f"Multiple MJDs found in '{s}'"
    return int(m[0][0]), int(m[0][1])

def is_dataset_id(s: str):
    # dataset_id ends in an mjd
    m = re.match(dataset_id_regex, s)
    return m is not None

if __name__ == "__main__":
    assert is_dataset_id("22B-222.sb42556485.eb43739655.60027.486694224535")

def is_scan_id(s: str):
    # scan_id ends in an mjd and then a enumerative_pair
    m = re.match(scan_id_regex, s)
    return m is not None

def split_path_up(s):
    steps_rev = []
    while True:
        head, tail = os.path.split(s)
        if len(tail) > 0:
            steps_rev.append(tail)
        if len(head) == 0 or head == s:
            break
        s = head
    steps_rev.reverse()
    return steps_rev

def filter_first(container, predicate):
    for el in container:
        if predicate(el):
            return el
    return None

if __name__ == "__main__":
    path_parts = split_path_up('/mnt/cosmic-storage-1/data0/vlass_target_discard/22B-222.sb42556485.eb43739655.60027.486694224535/TCOS0001_lie/TCOS0001_lie.AC.C672.0010.seticore.hits')
    dataset_id = filter_first(path_parts, is_dataset_id)
    assert dataset_id == "22B-222.sb42556485.eb43739655.60027.486694224535", f"{dataset_id} incorrect"

    path_parts = split_path_up('/mnt/cosmic-storage-1/data0/vlass_target/VLASS3.1.sb43319939.eb43644901.59991.14533251157.100.1/')
    scan_id = filter_first(path_parts, is_scan_id)
    assert scan_id == "VLASS3.1.sb43319939.eb43644901.59991.14533251157.100.1", f"{scan_id} incorrect"

    fp = '/mnt/buf1/vlass_target/GUPPI/VLASS4.1.sb49141666.eb49343448.60913.730115625.53.1.BD.C672.0005.raw.seticore.0000.stamps'
    frm = FilepathRegexMatch.from_filepath(fp)
    print(frm)
    path_parts = split_path_up('/mnt/buf1/vlass_target/GUPPI/VLASS4.1.sb49141666.eb49343448.60913.730115625.53.1.BD.C672.0005.raw.seticore.0000.stamps')
    scan_id = filter_first(path_parts, is_scan_id)
    assert scan_id == "VLASS4.1.sb49141666.eb49343448.60913.730115625.53.1", f"{scan_id} incorrect"

def find_path_at(
    regex, search_root,
    is_file_not_dir: bool = True
):
    at_this_level = []
    recurse = []
    # for root, dirs, files in os.walk(search_root, topdown=True):
    for entry in os.listdir(search_root):
        path = os.path.join(search_root, entry)
        if os.path.isdir(path):
            recurse.append(path)
            if not is_file_not_dir and re.match(regex, path) is not None:
                at_this_level.append(path)
            
        if not is_file_not_dir:
            continue

        if os.path.isfile(path):
            if re.match(regex, path) is not None:
                at_this_level.append(path)

    return at_this_level, recurse

def _find_path_default_error_handle(findings, err):
    raise RuntimeError(f"Failed on: {findings}") from err

def find_path(
    regex, search_roots: list, pool,
    callback,
    is_file_not_dir: bool = True,
    stop_earliest: bool = True,
    callback_error = _find_path_default_error_handle
):
    async_results = [
        pool.apply_async(
            find_path_at,
            (
                regex,
                root
            ),
            {
                "is_file_not_dir": is_file_not_dir
            }
        )
        for root in search_roots
    ]
    while len(async_results) > 0:
        i = 0
        i_lim = len(async_results)
        while i < i_lim:
            if not async_results[i].ready():
                i += 1
                continue

            i_lim -= 1 # ensure the older async get checked before the newly appended ones
            found, dirs_further = async_results.pop(i).get()
            if len(found) > 0:
                try:
                    callback(found)
                except BaseException as err:
                    callback_error(found, err)
                if stop_earliest:
                    continue

            if len(dirs_further) == 0:
                continue
            async_results.extend([
                pool.apply_async(
                    find_path_at,
                    (
                        regex,
                        dir_further
                    ),
                    {
                        "is_file_not_dir": is_file_not_dir
                    }
                )
                for dir_further in dirs_further
            ])