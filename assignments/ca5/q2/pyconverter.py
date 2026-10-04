from nbconvert import PythonExporter
import nbformat

# Load the notebook
with open('Q2.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(notebook)

# Save to .py file
with open('Q2.py', 'w') as f:
    f.write(source)
