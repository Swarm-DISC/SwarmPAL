from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.run(
        "uv",
        "sync",
        "--active",
        "--frozen",
        "--group",
        "test",
        "--group",
        "apexpy_wheels",
        "--extra",
        "experimental",
    )
    session.run("pytest", *session.posargs)


@nox.session(python="3.11")
def docs(session: nox.Session) -> None:
    """
    Build the docs. Fast mode (default) skips notebook execution and autoapi.

    Pass "--full" for a full build (notebook execution + autoapi).
    Pass "serve" to serve the built docs.

    e.g. uvx nox -s docs -- --full serve
    """

    session.run(
        "uv",
        "sync",
        "--active",
        "--frozen",
        "--group",
        "docs",
        "--group",
        "apexpy_wheels",
        "--extra",
        "experimental",
    )

    known_args = {"--full", "serve"}
    unknown = [a for a in session.posargs if a not in known_args]
    if unknown:
        session.warn(f"Unsupported argument(s) to docs: {unknown}")

    full = "--full" in session.posargs

    sphinx_args = ["-b", "html"]
    env = {}
    if not full:
        sphinx_args.extend(["-D", "nb_execution_mode=off"])
        env["FAST_DOCS"] = "1"

    sphinx_args.extend(["docs", "docs/_build/html"])

    session.run("sphinx-build", *sphinx_args, env=env)

    if "serve" in session.posargs:
        print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
        session.run("python", "-m", "http.server", "8000", "-d", "docs/_build/html")
