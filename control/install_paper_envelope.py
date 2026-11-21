"""Replace the old envelope callable in either independent package; retain backup."""
import ast
import hashlib
import json
from pathlib import Path
import sys


def install(root):
    root=Path(root).resolve()
    if root.name not in ('gaia_massive_star_circularization','massive_star_circularization'):
        raise ValueError('Expected an independent massive-star sampling package')
    path=root/'source/bayesian/windemuth_et_al_util.py'
    text=path.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id=='eccentricity_envelope' for t in n.targets)]
    if len(nodes)!=1:raise ValueError('Expected one envelope assignment')
    node=nodes[0];lines=text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['eccentricity_envelope = PaperEccentricityEnvelope()\n']
    updated=''.join(lines)
    imp='from bayesian.paper_eccentricity_envelope import PaperEccentricityEnvelope\n'
    if imp not in updated:updated=imp+updated
    ast.parse(updated)
    backup=path.with_name(path.name+'.before_paper_envelope_20261116')
    if not backup.exists():backup.write_text(text)
    path.write_text(updated)
    record=dict(package_root=str(root),envelope='e=min(.8,.06+2.154176170062391*log10(max(P,7)/7)); through Stock2; flat at.8 from15.438917056042d to50d and beyond',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'PAPER_ENVELOPE_UPDATE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':install(sys.argv[1])
