# LocalVideoTranslator

## Overview

- [Installation](#installation)

## Installation

1. Clone the repository:
    ```
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```
    cd <project-directory>
    ```

3. Create a virtual environment in Python:
    ```
    virtualenv --python=python3.10 <env_name>
    ```

4. Activate the environment and install the dependencies from the `requirements-dev.txt` file:
    ```
    source <env_name>
    pip install ".[dev]"
    ```

5. Install `pre-commit` before pushing any changes to git:
    ```
    pre-commit install
    ```
