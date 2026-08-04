First, create an environment
1. python -3.12 -m venv venv
2. .\venv\Scripts\Activate.ps1 

Then, install dependencies first:
  pip install -r requirements.txt

Run verify.py to check that the environment works. A screen with the lander should pop up, allowing the lander to start moving before the screen closes.


To run pytests, run:
  python -m pytest tests/