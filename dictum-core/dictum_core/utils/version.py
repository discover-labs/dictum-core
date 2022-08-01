from typing import List, Optional
import sys
from packaging import version
from dictum_core import __version__


def find_latest_compatible_version(target: str, candidates: List[str]) -> Optional[str]:
    target = version.parse(target)
    result = None

    for current in candidates:
        current = current.strip()
        ver = version.parse(current)
        if ver.major == target.major and ver.minor == target.minor:
            if result is None or version.parse(result) < ver:
                result = current

    return result


if __name__ == "__main__":
    target = __version__
    if len(sys.argv) > 1:
        target = sys.argv[1]
    result = find_latest_compatible_version(target, sys.stdin)
    print(result)
