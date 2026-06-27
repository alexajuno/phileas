"""``python -m phileas`` entry point, equivalent to the ``phileas`` console
script. Lets the capture hooks fall back to the running interpreter when the
console script isn't on PATH."""

from phileas.cli import app

if __name__ == "__main__":
    app()
