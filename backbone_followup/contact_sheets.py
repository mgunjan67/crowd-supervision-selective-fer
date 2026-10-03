"""QA contact sheets only; these are not submission illustrations."""
from pathlib import Path
import json
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BUILD=ROOT/'tmp/backbone_followup_pdf'
for kind in ('render','supplement-render'):
    report=json.loads((ROOT/'backbone_followup/build_report.json').read_text())
    count=report['pages'] if kind=='render' else report['supplement_pages']
    files=[BUILD/kind/f'page-{n:02d}.png' for n in range(1,count+1)];dest=BUILD/(kind+'-contacts');dest.mkdir(exist_ok=True)
    for start in range(0,len(files),4):
        canvas=Image.new('RGB',(1600,2310),'#cccccc');draw=ImageDraw.Draw(canvas)
        for j,p in enumerate(files[start:start+4]):
            with Image.open(p) as original:im=ImageOps.contain(original.convert('RGB'),(780,1110))
            x=(j%2)*800+10;y=(j//2)*1155+30;canvas.paste(im,(x,y));draw.text((x,y-20),kind+' '+p.stem,fill='black')
        canvas.save(dest/f'sheet-{start//4+1:02d}.png')
print('QA sheets prepared, manual inspection still required')
