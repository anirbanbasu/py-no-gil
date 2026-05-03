from pathlib import Path
import tomllib


def main() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_data = tomllib.load(pyproject_file)

    scripts = pyproject_data.get("project", {}).get("scripts", {})
    script_list = ", ".join(sorted(scripts)) if scripts else "(none)"
    print("Hello from Python with no GIL, a.k.a., png!")
    print(f"Available scripts: {script_list}")
