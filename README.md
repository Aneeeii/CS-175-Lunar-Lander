First, create an environment
1. python -3.12 -m venv venv
2. .\venv\Scripts\Activate.ps1 

Then, install dependencies first:
  pip install -r requirements.txt

Run verify.py to check that the environment works. A screen with the lander should pop up, allowing the lander to start moving before the screen closes.


To run pytests, run:
  python -m pytest tests/

Training/evaluation scaffold:
  python train.py --mode baseline
  python train.py --mode train
  python train.py --mode all

Example
  python train.py --mode all --baseline-episodes 5 --train-episodes 10 --eval-episodes 5