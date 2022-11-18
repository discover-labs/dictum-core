from packaging import version

from dictum_core import __version__

if __name__ == "__main__":
    ver = version.parse(__version__)
    print(f"{ver.major}.{ver.minor}")
