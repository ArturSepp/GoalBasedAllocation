"""Sphinx configuration for GoalBasedAllocation."""

from importlib.metadata import version as package_version

project = "GoalBasedAllocation"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
release = package_version("goal-based-allocation")
version = release

extensions = [
    "myst_parser",
    "sphinx_sitemap",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinxdoc"
html_title = "GoalBasedAllocation documentation"
html_short_title = "GoalBasedAllocation"
html_baseurl = "https://artursepp.github.io/GoalBasedAllocation/"
html_extra_path = ["robots.txt"]

sitemap_url_scheme = "{link}"
