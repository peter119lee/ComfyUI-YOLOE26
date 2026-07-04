# conftest.py — pytest root configuration
# This file marks the project root for pytest and prevents collection of
# the package __init__.py, which uses relative imports and is not a test module.
collect_ignore = ["__init__.py"]
