import sys, json, glob
from pathlib import Path

# Step B3 - Merge chunks
chunks = sorted(glob.glob(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_chunk_*.json'))
all_nodes, all_edges, all_hyperedges = [], [], []
total_in, total_out = 0, 0
for c in chunks:
    d = json.loads(Path(c).read_text(encoding='utf-8'))
    all_nodes += d.get('nodes', [])
    all_edges += d.get('edges', [])
    all_hyperedges += d.get('hyperedges', [])
    total_in += d.get('input_tokens', 0)
    total_out += d.get('output_tokens', 0)

Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_semantic_new.json').write_text(json.dumps({
    'nodes': all_nodes, 'edges': all_edges, 'hyperedges': all_hyperedges,
    'input_tokens': total_in, 'output_tokens': total_out,
}, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Merged {len(chunks)} chunks: {total_in:,} in / {total_out:,} out tokens')

# Cache new
from graphify.cache import save_semantic_cache
new = json.loads(Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_semantic_new.json').read_text(encoding='utf-8'))
saved = save_semantic_cache(new.get('nodes', []), new.get('edges', []), new.get('hyperedges', []), root=r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need')
print(f'Cached {saved} files')

# Merge cached + new
cached_path = Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_cached.json')
cached = json.loads(cached_path.read_text(encoding='utf-8')) if cached_path.exists() else {'nodes':[],'edges':[],'hyperedges':[]}
all_nodes = cached['nodes'] + new.get('nodes', [])
all_edges = cached['edges'] + new.get('edges', [])
all_hyperedges = cached.get('hyperedges', []) + new.get('hyperedges', [])
seen = set()
deduped = []
for n in all_nodes:
    if n['id'] not in seen:
        seen.add(n['id'])
        deduped.append(n)

merged_sem = {
    'nodes': deduped,
    'edges': all_edges,
    'hyperedges': all_hyperedges,
    'input_tokens': new.get('input_tokens', 0),
    'output_tokens': new.get('output_tokens', 0),
}
Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_semantic.json').write_text(json.dumps(merged_sem, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Extraction complete - {len(deduped)} nodes, {len(all_edges)} edges ({len(cached["nodes"])} from cache, {len(new.get("nodes",[]))} new)')

# Part C - Merge AST + semantic
ast = json.loads(Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_ast.json').read_text(encoding='utf-8'))
sem = merged_sem

seen = {n['id'] for n in ast['nodes']}
merged_nodes = list(ast['nodes'])
for n in sem['nodes']:
    if n['id'] not in seen:
        merged_nodes.append(n)
        seen.add(n['id'])

merged_edges = ast['edges'] + sem['edges']
merged_hyperedges = sem.get('hyperedges', [])
merged = {
    'nodes': merged_nodes,
    'edges': merged_edges,
    'hyperedges': merged_hyperedges,
    'input_tokens': sem.get('input_tokens', 0),
    'output_tokens': sem.get('output_tokens', 0),
}
Path(r'F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\graphify-out\.graphify_extract.json').write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Merged: {len(merged_nodes)} nodes, {len(merged_edges)} edges ({len(ast["nodes"])} AST + {len(sem["nodes"])} semantic)')
