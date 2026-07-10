import json
import traceback
from pathlib import Path
import sys

try:
    from graphify.detect import detect
    print("Graphify imported successfully.")
    
    result = detect(Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need'))
    
    out_path = Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_detect.json')
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print("Success. Wrote to file.")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
