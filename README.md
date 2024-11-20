# Setup for python scripts

1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install

2. Create a python environment.

    ```bash
    python3 -m venv ~/.virtualenvs/instructions
    ```

3. Install python dependencies.


    ```bash
    source ~/.virtualenvs/instructions/bin/activate
    pip install -r requirements.txt
    ```

4. Initialize the Google Cloud CLI.

    ```bash
    gcloud init
    gcloud auth application-default login
    ```

5. Connect Google CLI to our project ID

   ```bash
   project id = stations-243022
   time zone = us-central1
   ```
   
6. Run the python scripts.

    ```bash
    python3 gen_image.py
    python3 gen_text.py
    ```

# Setup for promptfoo tests

1. Download and install Node.js at https://nodejs.org/en/download/

2. Install Node.js dependencies.

    ```bash
    sudo npm install
    ```

3. Run the tests.

    ```bash
    promptfoo eval
    ```
