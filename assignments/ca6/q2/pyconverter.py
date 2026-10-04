from nbconvert import PythonExporter
import nbformat

input_file_name = input()
# Load the notebook
with open(f'{input_file_name}.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(notebook)

# Save to .py file
with open(f'{input_file_name}.py', 'w') as f:
    f.write(source)
