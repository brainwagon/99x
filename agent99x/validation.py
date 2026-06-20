import os
import subprocess
from typing import List, Optional


def validate_files(filepaths: List[str]) -> Optional[str]:
    """Run basic syntax validation on modified files. Return error string if failed."""
    errors = []
    for fp in filepaths:
        if not os.path.exists(fp):
            continue

        cmd = None
        if fp.endswith(".py"):
            cmd = ["python", "-m", "py_compile", fp]
        elif fp.endswith((".sh", ".bash")):
            cmd = ["bash", "-n", fp]
        elif fp.endswith(".c"):
            cmd = ["gcc", "-fsyntax-only", fp]
        elif fp.endswith((".cpp", ".cc", ".cxx", ".h", ".hpp")):
            cmd = ["g++", "-fsyntax-only", fp]
        elif fp.endswith(".js"):
            cmd = ["node", "--check", fp]
        elif fp.endswith(".json"):
            cmd = ["python", "-m", "json.tool", fp]
        elif fp.endswith((".yaml", ".yml")):
            cmd = ["python", "-c", "import sys, yaml; yaml.safe_load(open(sys.argv[1]))", fp]
        elif fp.endswith(".ini"):
            cmd = ["python", "-c", "import sys, configparser; c = configparser.ConfigParser(); c.read_file(open(sys.argv[1]))", fp]
        elif fp.endswith(".toml"):
            cmd = ["python", "-c", "import sys; sys.version_info >= (3, 11) and __import__('tomllib').load(open(sys.argv[1], 'rb'))", fp]
        elif fp.endswith((".xml", ".rss")):
            cmd = ["python", "-c", "import sys, xml.etree.ElementTree as ET; ET.parse(sys.argv[1])", fp]
        elif fp.endswith((".html", ".htm")):
            cmd = ["npx", "--no-install", "htmlhint", fp]
        elif fp.endswith(".css"):
            cmd = ["npx", "--no-install", "stylelint", fp]

        if cmd:
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                err_output = (e.stderr or "") + (e.stdout or "")
                errors.append(f"Syntax error in {fp}:\n{err_output.strip()}")
            except FileNotFoundError:
                # Tool not installed locally, gracefully skip
                pass

    if errors:
        return "\n\n".join(errors)
    return None
