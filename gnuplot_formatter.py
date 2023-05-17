
"""
Internal use. No documentation.
"""

file = "avgTC.dat"

with open(file, "r") as f:
    lines = f.readlines()

with open(file, "w") as f:
    for i, line in enumerate(lines):
        # Zeilennummer + Einschub
        new_line = str(i+10001) + "\t" + line
        f.write(new_line)
