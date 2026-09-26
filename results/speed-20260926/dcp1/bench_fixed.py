"""Fixed-input diagnostic wrapper; original benchmark stays unchanged."""
from pathlib import Path
import hashlib,json,random,runpy,string,sys
BENCH=Path('<workspace>/trellismx-performance-audit-20260908/llm_decode_bench.py')
EXPECTED='7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a'
assert hashlib.sha256(BENCH.read_bytes()).hexdigest()==EXPECTED
RUN_ID='tmxrepeatabc'
original_choices=random.choices
uses=0
def fixed_choices(population,weights=None,*,cum_weights=None,k=1):
    global uses
    if population==string.ascii_lowercase and k==12 and weights is None and cum_weights is None:
        uses+=1
        return list(RUN_ID)
    return original_choices(population,weights,cum_weights=cum_weights,k=k)
help_requested='--help' in sys.argv
if not help_requested:
    assert '--temperature' in sys.argv
    assert float(sys.argv[sys.argv.index('--temperature')+1])==0.0
random.choices=fixed_choices
print(json.dumps({'diagnostic_wrapper':'fixed-prefix-greedy','benchmark_sha256':EXPECTED,'run_id':RUN_ID,'temperature':0.0,'scope':'Separate diagnostic, not model-default throughput'}),flush=True)
sys.argv[0]=str(BENCH)
try:
    runpy.run_path(str(BENCH),run_name='__main__')
finally:
    random.choices=original_choices
    if not help_requested:
        print(json.dumps({'fixed_run_id_calls':uses,'benchmark_unchanged':hashlib.sha256(BENCH.read_bytes()).hexdigest()==EXPECTED}),flush=True)
        assert uses==1,uses
