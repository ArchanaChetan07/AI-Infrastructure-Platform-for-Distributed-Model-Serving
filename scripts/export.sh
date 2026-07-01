#!/usr/bin/env bash
set -euo pipefail
python -c "
from predictor.export import export_model
from pathlib import Path
import json, numpy as np
meta = Path('models/metadata.json')
if not Path('models/output_length_mlp.pt').exists():
    raise SystemExit('Train model first')
m = json.loads(meta.read_text()) if meta.exists() else {}
export_model('models/output_length_mlp.pt', 'models',
    np.array(m.get('feature_mean', [0]*40)),
    np.array(m.get('feature_std', [1]*40)))
print('Export complete.')
"
