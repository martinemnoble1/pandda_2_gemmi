"""Console-script entry point for ``pandda2.analyse``.

This lives inside the installed ``pandda_gemmi`` package so it is importable from
a ``console_scripts`` entry point. That fixes three problems with the previous
``scripts/pandda2.analyse`` bash wrapper:

* the wrapper referenced ``pandda.py``, which was never installed into ``bin/``;
* it used ``readlink -f``, which is unsupported by stock BSD ``readlink`` on macOS;
* it called a bare ``python3.9``, which on macOS resolves to Homebrew's
  interpreter ahead of the active conda env.

A ``console_scripts`` entry point is installed with the correct interpreter
shebang automatically and needs no path resolution, fixing all three.
"""

from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda.pandda import pandda


def main():
    # Parse command line arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)


if __name__ == '__main__':
    main()
