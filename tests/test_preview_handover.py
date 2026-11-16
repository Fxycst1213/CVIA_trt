"""Verify automatic handover cannot terminate another checkout or serial control."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PreviewHandoverTests(unittest.TestCase):
    def test_only_own_preview_processes_are_stopped(self):
        processes = []
        with tempfile.TemporaryDirectory(prefix="cvia-handover-") as directory:
            project = Path(directory) / "project"
            other = Path(directory) / "other"
            for target in (project, other):
                (target / "tools").mkdir(parents=True)
                (target / "tools/server.py").write_text("import time\ntime.sleep(60)\n")
            shutil.copy2(ROOT / "tools/stop_preview.py", project / "tools/stop_preview.py")
            subprocess.run(["cc", "-x", "c", "-o", str(project / "trt"), "-"],
                           input="#include <unistd.h>\nint main(void) { for (;;) pause(); }\n",
                           text=True, check=True, capture_output=True)
            shutil.copy2(project / "trt", other / "trt")
            (project / "start_preview.sh").write_text("trap 'exit 0' TERM\nwhile sleep 0.1; do :; done\n")

            def start(command, cwd=project):
                process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(process)
                return process

            try:
                own_camera = start(["./trt", "--preview", "--no-serial"])
                own_bridge = start([sys.executable, "tools/server.py", "--no-display"])
                own_launcher = start(["bash", "./start_preview.sh"])
                control_camera = start(["./trt", "--preview"])  # Serial control is not disabled.
                rtsp_bridge = start([sys.executable, "tools/server.py", "--rtsp-url", "rtsp://127.0.0.1/live"])
                rtsp_equals = start([sys.executable, "tools/server.py", "--rtsp-url=rtsp://127.0.0.1/live"])
                other_camera = start(["./trt", "--preview", "--no-serial"], other)
                other_bridge = start([sys.executable, "tools/server.py"], other)
                time.sleep(.2)
                for process in processes:
                    self.assertIsNone(process.poll())
                result = subprocess.run([sys.executable, "tools/stop_preview.py"], cwd=project,
                                        capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stderr)
                for process in (own_camera, own_bridge, own_launcher):
                    self.assertIsNotNone(process.poll())
                for process in (control_camera, rtsp_bridge, rtsp_equals, other_camera, other_bridge):
                    self.assertIsNone(process.poll())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
