import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

Coord = Tuple[int, int]


class PBSWrapper:
    def __init__(self, pbs_path: Optional[str] = None) -> None:
        self.pbs_path = self._resolve_pbs_path(pbs_path)

    @staticmethod
    def _is_wsl() -> bool:
        """Detect if running in WSL."""
        try:
            with open("/proc/version", "r") as f:
                content = f.read().lower()
                return "microsoft" in content or "wsl" in content
        except (FileNotFoundError, IOError):
            return False

    @staticmethod
    def _resolve_pbs_path(pbs_path: Optional[str]) -> str:
        """Resolve PBS executable path robustly across Windows/WSL/Unix."""
        is_windows = sys.platform == "win32"
        is_wsl = PBSWrapper._is_wsl()

        # Explicit path provided
        if pbs_path:
            candidate = Path(pbs_path)
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())

        repo_root = Path(__file__).resolve().parent

        # WSL: prefer Unix binary, skip .exe
        if is_wsl:
            candidates = [repo_root / "pbs"]
        # Windows: prefer .exe over Unix binary
        elif is_windows:
            candidates = [repo_root / "pbs.exe", repo_root / "pbs"]
        # Unix/Linux: prefer Unix binary
        else:
            candidates = [repo_root / "pbs", repo_root / "pbs.exe"]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                # On Windows, skip Unix binaries (no extension)
                if is_windows and candidate.suffix == "":
                    continue
                return str(candidate.resolve())

        # Try system PATH
        installed = shutil.which("pbs")
        if installed:
            return installed

        # Special case: on Windows, more helpful error
        if is_windows:
            raise FileNotFoundError(
                f"PBS executable for Windows not found at {repo_root / 'pbs.exe'}.\n"
                f"To run on Windows, build pbs.exe: cmake -DCMAKE_BUILD_TYPE=RELEASE . && cmake --build . --config Release\n"
                f"Or run on WSL: wsl python3 {os.path.abspath(__file__)}"
            )

        # Fallback: return sensible default for error message
        raise FileNotFoundError(
            f"PBS executable not found. Tried:\n"
            f"  - {pbs_path or 'provided path'}\n"
            f"  - {repo_root / 'pbs'}\n"
            f"  - {repo_root / 'pbs.exe'}\n"
            f"  - system PATH\n"
            f"Build PBS first: cmake -DCMAKE_BUILD_TYPE=RELEASE . && make"
        )

    @staticmethod
    def _validate_file(path: str, expected_suffix: str, label: str) -> None:
        p = Path(path)
        if p.suffix.lower() != expected_suffix:
            raise ValueError(f"Expected {label} with {expected_suffix} extension, got: {path}")
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    @staticmethod
    def parse_paths(path_file: str) -> List[List[Coord]]:
        """Parse coordinate paths from PBS output."""
        result: List[List[Coord]] = []
        pair_pattern = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

        with open(path_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                coords = [(int(x), int(y)) for x, y in pair_pattern.findall(line)]
                if coords:
                    result.append(coords)

        return result

    def solve(
        self,
        map_file: str,
        scen_file: str,
        out_csv: str,
        output_paths: str,
        k: int,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Solve MAPF using PBS using the same argument order as PBS CLI:
          map_file, scen_file, out_csv, output_paths, k, timeout

        Returns dict with keys:
          - success: bool
          - paths: List[List[Coord]] or None
          - error: str (if success=False)
          - returncode: int (PBS exit code)
          - stdout/stderr: str (if available)
        """
        try:
            self._validate_file(map_file, ".map", "Map file")
            self._validate_file(scen_file, ".scen", "Scenario file")
            if k <= 0:
                raise ValueError(f"k must be > 0, got: {k}")
            if timeout <= 0:
                raise ValueError(f"timeout must be > 0, got: {timeout}")
        except ValueError as e:
            return {
                "success": False,
                "paths": None,
                "error": f"Input validation failed: {e}",
                "returncode": None,
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "paths": None,
                "error": str(e),
                "returncode": None,
            }

        cmd = [
            self.pbs_path,
            "-m", map_file,
            "-a", scen_file,
            "-o", out_csv,
            "--outputPaths", output_paths,
            "-k", str(k),
            "-t", str(timeout),
        ]

        try:
            run = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "paths": None,
                "error": f"PBS timeout after {timeout} seconds",
                "returncode": None,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "paths": None,
                "error": f"PBS executable not found: {self.pbs_path}",
                "returncode": None,
            }
        except OSError as e:
            return {
                "success": False,
                "paths": None,
                "error": f"Failed to run PBS: {e}",
                "returncode": None,
            }

        stdout = run.stdout or ""
        stderr = run.stderr or ""

        if run.returncode != 0:
            return {
                "success": False,
                "paths": None,
                "error": f"PBS exited with code {run.returncode}",
                "returncode": run.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }

        if not os.path.exists(output_paths):
            return {
                "success": False,
                "paths": [],
                "warning": "PBS completed without writing a paths file",
                "returncode": run.returncode,
                "stdout": stdout,
            }

        if os.path.getsize(output_paths) == 0:
            return {
                "success": False,
                "paths": [],
                "warning": "PBS completed but the paths file was empty",
                "returncode": run.returncode,
                "stdout": stdout,
            }

        try:
            paths = self.parse_paths(output_paths)
            return {
                "success": True,
                "paths": paths,
                "returncode": run.returncode,
                "stdout": stdout,
            }
        except Exception as e:
            return {
                "success": False,
                "paths": None,
                "error": f"Path parsing failed: {e}",
                "returncode": run.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }


def solve(
    map_file: str,
    scen_file: str,
    out_csv: str,
    output_paths: str,
    k: int,
    timeout: int,
    pbs_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper with the same argument order as PBS CLI."""
    try:
        wrapper = PBSWrapper(pbs_path=pbs_path)
    except FileNotFoundError as e:
        return {
            "success": False,
            "paths": None,
            "error": str(e),
            "returncode": None,
        }

    return wrapper.solve(
        map_file=map_file,
        scen_file=scen_file,
        out_csv=out_csv,
        output_paths=output_paths,
        k=k,
        timeout=timeout,
    )


if __name__ == "__main__":
    if len(sys.argv) not in (7, 8):
        print(
            "Usage: python pbs_wrapper.py <map_file> <scen_file> <out_csv> <output_paths> <k> <timeout> [pbs_path]"
        )
        sys.exit(1)

    map_file_arg = sys.argv[1]
    scen_file_arg = sys.argv[2]
    out_csv_arg = sys.argv[3]
    output_paths_arg = sys.argv[4]

    try:
        k_arg = int(sys.argv[5])
        timeout_arg = int(sys.argv[6])
    except ValueError:
        print("k and timeout must be integers")
        sys.exit(1)

    pbs_path_arg = sys.argv[7] if len(sys.argv) == 8 else None

    result = solve(
        map_file=map_file_arg,
        scen_file=scen_file_arg,
        out_csv=out_csv_arg,
        output_paths=output_paths_arg,
        k=k_arg,
        timeout=timeout_arg,
        pbs_path=pbs_path_arg,
    )

    if result.get("success"):
        print("PBS run succeeded")
        sys.exit(0)

    print(result.get("error", "PBS run failed"))
    sys.exit(1)
