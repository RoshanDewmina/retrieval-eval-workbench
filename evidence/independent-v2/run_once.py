"""Execute the frozen independent set once; never tune or silently retry."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / 'retrieval-eval-workbench'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert revision == manifest['frozen_implementation_revision'], 'Implementation revision drift'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip(), 'Implementation is dirty'
    for filename, expected in manifest['files'].items():
        assert sha(HERE / filename) == expected, filename
    for filename, expected in manifest['input_fingerprints'].items():
        assert sha(REPO / filename) == expected, filename
    freeze = json.loads((HERE / 'freeze.json').read_text())
    assert sha(HERE / 'manifest.json') == freeze['manifest_sha256']
    assert sha(Path(__file__)) == freeze['runner_sha256']
    with (HERE / 'run-marker.json').open('x') as stream:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'revision': revision,
                   'freeze_sha256': sha(HERE / 'freeze.json')}, stream, indent=2)
    import torch
    torch.set_num_threads(2)
    from retrieval_eval_workbench.data import Question, ExpectedCitation, load_documents, load_frozen_settings, load_model_manifest
    from retrieval_eval_workbench.retrieval import LexicalRetriever, SemanticRetriever
    from retrieval_eval_workbench.evaluation import evaluate
    docs = load_documents()
    settings = load_frozen_settings()
    model = load_model_manifest()
    labels = json.loads((HERE / 'labels.json').read_text())['labels']
    questions = [Question(**{**q, 'relevant_document_ids': tuple(q['relevant_document_ids']),
        'supporting_citations': tuple(ExpectedCitation(**c) for c in labels[q['id']]['supporting_citations']),
        'required_answer_phrases': tuple(labels[q['id']]['required_answer_phrases'])})
        for q in json.loads((HERE / 'questions.json').read_text())['questions']]
    result = {'schema_version': 2, 'source_revision': revision, 'dirty_tree': False,
              'command': 'uv run --frozen python3 ../docs/reviews/retrieval-fresh-holdout/run_once.py',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'exit_status': None,
              'environment': {'os': platform.platform(), 'python': sys.version, 'architecture': platform.machine(),
                              'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads(),
                              'cpu_count': os.cpu_count(), 'offline': True},
              'freeze_sha256': sha(HERE / 'freeze.json'), 'manifest_sha256': sha(HERE / 'manifest.json'),
              'model': model, 'settings': settings, 'results': {}, 'supplemental': {},
              'measurement_kind': 'one_actual_local_pass_after_independent_question_and_code_freeze',
              'external_cost_usd': 0, 'limitations': manifest['limitations'] + manifest['metric_semantics']}
    destination = HERE / 'first-run.json'
    try:
        for key, cls in [('lexical', LexicalRetriever), ('semantic', SemanticRetriever)]:
            began = time.monotonic()
            retriever = cls(docs)
            initialized = time.monotonic()
            measured = evaluate(retriever, questions, docs, split='test', limit=settings['top_k'],
                                threshold=settings['retrievers'][key]['minimum_score'])
            result['results'][key] = measured
            multi = [row for row in measured['examples'] if manifest['categories'][row['question_id']] == 'multi_document']
            result['supplemental'][key] = {
                'initialization_ms': round((initialized - began) * 1000, 3),
                'evaluation_ms': round((time.monotonic() - initialized) * 1000, 3),
                'multi_document_cases': [{'question_id': row['question_id'],
                     'all_required_docs_retrieved_at_3': set(row['expected_document_ids']).issubset(row['ranked_document_ids']),
                     'fraction_required_docs_retrieved_at_3': len(set(row['expected_document_ids']) & set(row['ranked_document_ids'])) / len(row['expected_document_ids']),
                     'complete_critical_facts': row['critical_fact']} for row in multi]}
            destination.write_text(json.dumps(result, indent=2) + '\n')
            print(key, json.dumps(measured['metrics']), flush=True)
        result['exit_status'] = 0
    except Exception as exc:
        result['exit_status'] = 1
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    main()
