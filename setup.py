import os
from distutils.core import setup
from typing import List
from Cython.Build import cythonize
from Cython.Compiler import Options


# comile cmd
# python3.7 core/setup.py build_ext --inplace

py_files: List[str] = ["./test/utils.py"]
c_files: List[str] = [file.replace(".py", ".c") for file in py_files]

Options.docstrings = False
setup(
    ext_modules=cythonize(py_files, exclude="setup.py",
                          compiler_directives={'language_level': "3"}),
    name="utils",
)

for file in c_files:
    try:
        os.remove(file)
    except FileNotFoundError:
        print("skip file:", file)