from __future__ import annotations

import shutil
from pathlib import Path

import nox
import nox_uv

nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox_uv.session(python=["3.10", "3.11"], uv_groups=["test"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[dsecs,experimental]")
    session.run("pytest", *session.posargs)


@nox_uv.session(python="3.11", uv_groups=["docs"])
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "serve" to serve, "no-exec" to skip notebook execution.

    e.g. uvx nox -s docs -- no-exec
    """

    session.install(".[dsecs,experimental]")

    sphinx_args = ["-b", "html"]

    # Add notebook execution mode override if requested
    if "no-exec" in session.posargs:
        sphinx_args.extend(["-D", "nb_execution_mode=off"])

    sphinx_args.extend(["docs", "docs/_build/html"])

    session.run("sphinx-build", *sphinx_args)

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "docs/_build/html")
        elif "no-exec" not in session.posargs:
            session.warn("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_p = DIR.joinpath("build")
    if build_p.exists():
        shutil.rmtree(build_p)

    session.install("build")
    session.run("python", "-m", "build")
