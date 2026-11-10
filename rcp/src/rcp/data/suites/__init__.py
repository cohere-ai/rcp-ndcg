"""The four shipped suite definitions, as data: ``<name>.yaml``, read by :mod:`rcp.data.io.suite`.

A suite is data, not code: a downstream adds a benchmark by writing the same file in its own directory and
loading ``suite:<path>``. These four are the shipped view (:data:`rcp.data.SUITES`) and the source
``load_dataset("suite:<name>")`` reads.
"""

__all__: list[str] = []
