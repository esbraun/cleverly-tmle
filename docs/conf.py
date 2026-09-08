"""Sphinx configuration for the cleverly documentation site."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleverly import __version__  # noqa: E402

project = "cleverly"
author = "cleverly contributors"
copyright = "2026, cleverly contributors"
version = __version__
release = __version__

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

root_doc = "index"
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "api/generated/*.md", "Thumbs.db", ".DS_Store"]

# Notebooks are executed deliberately before review and commit their outputs.  Documentation
# builds remain deterministic, offline, and quick: they render those stored outputs without
# downloading data or refitting estimators.
nb_execution_mode = "off"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autosummary_generate = True
# Types belong in the signature, not the description.  numpydoc renders a docstring as one
# ``:Parameters:`` field rather than one ``:param x:`` field per parameter, so autodoc finds
# nothing to merge its annotations into and appends a second, description-less Parameters block.
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autoclass_content = "both"

# Public class pages expose the methods and attributes that the class defines.  Inherited
# container helpers stay on their defining class so result pages do not fill with Mapping methods.
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
# The build fails on a docstring that is structurally incomplete, because `pages.yml`
# builds with `-W`.  What is deliberately *off*: GL01, because a summary on the opening-quote
# line is house style here; ES01, because an extended summary is not owed by every one-line
# accessor; and EX01 and SA01, because Examples and See Also are required on the task spine
# only.  `tests/unit/test_documentation_api.py` enforces those two, since numpydoc's Sphinx
# configuration can require a check of every object or of none, and 140 pro-forma
# constructor examples are noise.
numpydoc_validation_checks = {
    "GL06",  # known section names
    "GL07",  # standard section order
    "PR01",  # every parameter is documented
    "PR02",  # no documented parameter that the signature does not have
    "PR04",  # every parameter carries a type
    "PR10",  # the "name : type" form, which numpydoc needs to split the two
    "RT01",  # a function that returns something says what
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "cleverly: the Python toolbox for TMLE"
html_baseurl = "https://esbraun.github.io/cleverly-tmle/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_theme_options = {
    "github_url": "https://github.com/esbraun/cleverly-tmle",
    # The navbar brand does not shrink or wrap. The full html_title is 329 px
    # wide, which pushes the header links onto a second row under about 1350 px
    # and leaves the brand overlapping the first link.
    "logo": {"text": "cleverly"},
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "esbraun",
    "github_repo": "cleverly-tmle",
    "github_version": "main",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d+\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True

# The ``docs`` job and ``pages.yml`` build with ``-W``, so one warning fails the run.
# ``sphinx.ext.intersphinx`` warns when it cannot fetch an inventory, which makes the gate
# depend on four third-party sites staying reachable.  A ``docs.scipy.org`` outage is not a
# defect in this repository, and an unresolved cross-reference still renders as plain text.
# The warning carries no type, so ``suppress_warnings`` cannot name it.  Drop that one
# message instead, and keep every other warning fatal.
_UNREACHABLE_INVENTORIES = "failed to reach any of the inventories"


class _ReachableInventoryFilter(logging.Filter):
    """Drop the one intersphinx warning that a third-party outage raises."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` for the unreachable-inventory warning.

        Parameters
        ----------
        record : logging.LogRecord
            Record Sphinx is about to emit.

        Returns
        -------
        bool
            ``True`` to keep the record, and ``False`` to drop it.
        """
        return _UNREACHABLE_INVENTORIES not in record.getMessage()


def setup(app: Any) -> None:  # numpydoc ignore=PR01
    """Register the inventory filter ahead of the warning-as-error filter.

    Sphinx installs ``WarningIsErrorFilter`` on the warning handler. Filters run in the
    order they appear, so this one goes in front of it. Appending would let ``-W`` raise
    before the record is dropped.
    """
    log_filter = _ReachableInventoryFilter()
    sphinx_logger = logging.getLogger("sphinx")
    sphinx_logger.addFilter(log_filter)
    for handler in sphinx_logger.handlers:
        handler.filters.insert(0, log_filter)
