#!/usr/bin/env python3
"""
Script to automatically generate Sphinx documentation for all modules and build the HTML website.
"""
import importlib.util
import os
import subprocess
import sys


def check_and_install_dependencies():
    """Check for required dependencies and install them if missing."""
    required_packages = [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinxcontrib-napoleon",
        "sphinxcontrib-mermaid",
        "sphinx-autodoc-typehints",
    ]

    missing_packages = []

    for package in required_packages:
        # Convert package name to module name (replace - with _)
        module_name = package.replace("-", "_")

        # Check if the package is installed
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package)

    # Install missing packages
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing_packages
        )
        print("Dependencies installed successfully")
    else:
        print("All required dependencies are already installed")


def create_makefile(docs_dir):
    """Create a Makefile for Sphinx documentation if it doesn't exist."""
    makefile_path = os.path.join(docs_dir, "Makefile")

    if os.path.exists(makefile_path):
        print(f"Makefile already exists at {makefile_path}")
        return

    print(f"Creating Makefile at {makefile_path}")

    makefile_content = """# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXFLAGS)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(SPHINXFLAGS)
"""

    with open(makefile_path, "w") as f:
        f.write(makefile_content)

    print("Makefile created successfully")


def create_make_bat(docs_dir):
    """Create a make.bat file for Windows if it doesn't exist."""
    make_bat_path = os.path.join(docs_dir, "make.bat")

    if os.path.exists(make_bat_path):
        print(f"make.bat already exists at {make_bat_path}")
        return

    print(f"Creating make.bat at {make_bat_path}")

    make_bat_content = """@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
"""

    with open(make_bat_path, "w") as f:
        f.write(make_bat_content)

    print("make.bat created successfully")


def main():
    # Check and install required dependencies
    print("=== Checking dependencies ===")
    check_and_install_dependencies()

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the project root
    project_root = os.path.dirname(script_dir)

    # Path to the source directory
    source_dir = os.path.join(project_root, "src")

    # Path to the docs source directory
    docs_source_dir = os.path.join(script_dir, "source")

    # Print paths for debugging
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Source directory: {source_dir}")
    print(f"Docs source directory: {docs_source_dir}")

    # Make sure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist!")
        sys.exit(1)

    # Make sure the docs source directory exists
    if not os.path.exists(docs_source_dir):
        print(f"Creating docs source directory: {docs_source_dir}")
        os.makedirs(docs_source_dir)

    # Step 1: Run sphinx-apidoc to generate .rst files for all modules
    print("\n=== Generating API documentation ===")
    cmd = [
        "sphinx-apidoc",
        "-f",  # Force overwriting of existing files
        "-e",  # Put module documentation before submodule documentation
        "-M",  # Put module documentation before subpackage documentation
        "-o",
        docs_source_dir,  # Output directory
        source_dir,  # Source code directory
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print the output of the command
    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        print(f"Error: sphinx-apidoc failed with return code {result.returncode}")
        sys.exit(1)

    # List the files in the docs source directory
    print("\nFiles in docs/source directory:")
    for file in sorted(os.listdir(docs_source_dir)):
        print(f"  {file}")

    print("\nDocumentation source files generated successfully!")

    # Step 2: Create Makefile and make.bat if they don't exist
    create_makefile(script_dir)
    create_make_bat(script_dir)

    # Step 3: Build the HTML documentation
    print("\n=== Building HTML documentation ===")

    # Determine the build command based on the platform
    if os.name == "nt":  # Windows
        build_cmd = ["make.bat", "html"]
    else:  # Unix/Linux/Mac
        build_cmd = ["make", "html"]

    # Change to the docs directory to run the build command
    os.chdir(script_dir)

    print(f"Running command: {' '.join(build_cmd)}")
    build_result = subprocess.run(build_cmd, capture_output=True, text=True)

    # Print the output of the build command
    print("STDOUT:")
    print(build_result.stdout)

    print("STDERR:")
    print(build_result.stderr)

    if build_result.returncode != 0:
        print(f"Error: HTML build failed with return code {build_result.returncode}")
        sys.exit(1)

    # Get the path to the built HTML documentation
    html_dir = os.path.join(script_dir, "build", "html")
    index_path = os.path.join(html_dir, "index.html")

    if os.path.exists(index_path):
        print(f"\nHTML documentation built successfully!")
        print(f"You can view it by opening: {index_path}")

        # Try to open the documentation in a browser
        try:
            import webbrowser

            print("\nAttempting to open documentation in your default browser...")
            webbrowser.open(f"file://{index_path}")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
    else:
        print(f"\nWarning: HTML index file not found at {index_path}")


if __name__ == "__main__":
    main()
