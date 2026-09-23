import json
from pathlib import Path
import unittest
import nbformat

ROOT = Path(__file__).resolve().parents[1]

class NotebookTests(unittest.TestCase):
    def test_schema_and_syntax(self):
        for path in (ROOT / "Tutorials").glob("*.ipynb"):
            with self.subTest(path=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                ids = [c.id for c in notebook.cells]
                self.assertEqual(len(ids), len(set(ids)))
                for i, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        compile(cell.source, f"{path.name}:cell{i}", "exec")
                        self.assertEqual(cell.outputs, [])

    def test_quickstart_headless(self):
        notebook = nbformat.read(ROOT / "Tutorials/Tutorial0-Quickstart.ipynb", as_version=4)
        scope = {"__name__": "notebook_validation"}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"Tutorial0:cell{i}", "exec"), scope)

if __name__ == "__main__":
    unittest.main()
