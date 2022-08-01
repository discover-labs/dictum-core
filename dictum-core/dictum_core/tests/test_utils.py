from dictum_core.utils.version import find_latest_compatible_version


def test_find_version():
    assert (
        find_latest_compatible_version(
            "0.1.0",
            ["0.0\n", "0.1.0-dev123\n", "0.1.2-dev3\n", "0.1.14-dev1\n", "0.4.18\n"],
        )
        == "0.1.14-dev1"
    )
