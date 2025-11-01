
import os
import tarfile
from pathlib import Path
from typing import List

def list_files(path: str) -> List[str]:
    """
    Lists all filenames within a folder — works for both normal folders
    and folders inside a .tar archive.

    Supports syntax:
        /data/archive.tar!/a/b/c/
    or
        /a/b/c/

    Parameters
    ----------
    path : str
        Folder path, either a real directory or a folder inside a .tar archive.

    Returns
    -------
    files : List[str]
        List of filenames (relative to the folder given).
    """
    # Check if this is a path inside a tar archive
    if ".tar" in path:
        # Split the tar archive path and the internal directory
        tar_path, inner_path = path.split(".tar", 1)
        tar_path = tar_path + ".tar"
        inner_path = inner_path.lstrip("!/")
        
        if not Path(tar_path).is_file():
            raise FileNotFoundError(f"Tar archive not found: {tar_path}")

        files = []
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                # Only list files under the specified internal folder
                if member.isfile() and member.name.startswith(inner_path):
                    rel_name = os.path.relpath(member.name, inner_path)
                    # Skip files in parent directories
                    if not rel_name.startswith(".."):
                        files.append(rel_name)
        return files

    else:
        # Normal folder case
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Not a valid directory: {path}")

        return [
            os.path.relpath(os.path.join(root, file), start=path)
            for root, _, files in os.walk(path)
            for file in files
        ]


def read_file_set(path: str):
    """
    Reads lines from a text file, either from a regular file or from within a .tar archive.
    
    Supports tar paths like:
        /path/to/archive.tar!/a/b/c/file.txt

    Parameters
    ----------
    path : str
        Path to a text file, or to a file within a tar archive using the 'tar!' notation.

    Returns
    -------
    file_set : List[str]
        List of stripped lines from the text file.
    """
    # Detect if we're reading from inside a .tar
    if ".tar" in path:
        # Example: /data/archive.tar!/a/b/c/file.txt
        tar_path, inner_path = path.split(".tar", 1)
        tar_path = tar_path + ".tar"
        inner_path = inner_path.lstrip("!/")
        
        if not Path(tar_path).is_file():
            raise FileNotFoundError(f"Tar archive not found: {tar_path}")

        with tarfile.open(tar_path, "r:*") as tar:
            try:
                member = tar.getmember(inner_path)
            except KeyError:
                raise FileNotFoundError(f"File '{inner_path}' not found inside {tar_path}")

            f = tar.extractfile(member)
            if f is None:
                raise IOError(f"Unable to extract {inner_path} from {tar_path}")

            lines = f.read().decode("utf-8").splitlines()
            return [line.strip() for line in lines]

    # Otherwise, it's a normal file on disk
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [x.strip() for x in f.readlines()]
