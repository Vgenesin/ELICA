project = "ELiCA"
copyright = "2024, Genesini V., Galloni G., Pagano L."
author = "Genesini V., Galloni G., Pagano L."

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cobaya": ("https://cobaya.readthedocs.io/en/latest/", None),
}

autodoc_member_order = "bysource"
