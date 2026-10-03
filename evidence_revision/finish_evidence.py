"""Resume-safe numerical pipeline; never evaluates unfinished training phases."""
import subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def main():
    while not (OUT/'training_complete.json').exists():time.sleep(10)
    for script in ('evaluate.py','analysis.py','audit.py'):
        subprocess.run([sys.executable,str(OUT/script)],check=True)
    print('NUMERICAL REVISION COMPLETE; MANUSCRIPT BUILD AND VISUAL REVIEW STILL REQUIRED',flush=True)
if __name__=='__main__':main()
