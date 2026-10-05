from imml import octavemodule_installed

ENGINES = ["python"]
if octavemodule_installed:
    ENGINES += ["octave"]

