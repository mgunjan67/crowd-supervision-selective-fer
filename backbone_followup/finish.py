"""Wait for all frozen training, then evaluate/audit/build, never inspect outcomes early."""
import subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def main():
    while not (OUT/'training_complete.json').exists():time.sleep(10)
    for script in ('evaluate.py','analysis.py','audit.py','audit_targets.py'):
        subprocess.run([sys.executable,str(OUT/script)],check=True)
    runtime=Path('C:/Users/mgunj/.cache/codex-runtimes/codex-primary-runtime')
    subprocess.run([str(runtime/'dependencies/node/bin/node.exe'),str(runtime/'plugins/openai-primary-runtime/plugins/pdf/skills/pdf/container_tools/mark_artifact_operation_started.mjs'),'--operation-kind','edit','--expected-output-count','2','--output-format','pdf'],check=True)
    for script in ('build_revision.py','build.py','write_handoff.py'):
        subprocess.run([sys.executable,str(OUT/script)],check=True)
    print('Numerical audit and PDF rendering completed; manual visual verification and packaging remain',flush=True)
if __name__=='__main__':main()
