"""
This Bash script is used to install the required packages for the project.

The script will install the following packages:

- Latest version of Python.

- Required Python packages from the requirements.txt file.
"""

python --version
pip install -r requirements.txt

# chmod +x install_requirements.sh
# ./install_requirements.sh