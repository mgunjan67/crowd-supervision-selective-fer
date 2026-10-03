"""Build after completed numerical audits; visual approval stays manual."""
import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def main():
    while not (OUT/'audit.json').exists():time.sleep(10)
    assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
    runtime=Path('C:/Users/mgunj/.cache/codex-runtimes/codex-primary-runtime')
    subprocess.run([str(runtime/'dependencies/node/bin/node.exe'),str(runtime/'plugins/openai-primary-runtime/plugins/pdf/skills/pdf/container_tools/mark_artifact_operation_started.mjs'),'--operation-kind','edit','--expected-output-count','2','--output-format','pdf'],check=True)
    for script in ('build_revision.py','build.py','write_handoff.py'):
        subprocess.run([sys.executable,str(OUT/script)],check=True)
    print('PDFS COMPILED AND RENDERED; VISUAL APPROVAL AND PUBLICATION NOT YET PERFORMED',flush=True)
if __name__=='__main__':main()
