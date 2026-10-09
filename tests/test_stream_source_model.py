"""Source-based stream ordering regression tests; no GPU execution."""
import tempfile
import unittest
from pathlib import Path
from cuda_source_model import analyze, UnsupportedSource

ROOT = Path(__file__).resolve().parents[1]

class StreamSourceTests(unittest.TestCase):
    def analyze_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'renamed.cu'
            path.write_text(text)
            return analyze(path)

    def test_stream_graph_and_source_changes(self):
        source = (ROOT / 'workloads/fork_join_streams.cu').read_text()
        source = source.replace('(1 << 20) + 17', '513').replace('addStream', 'workerA').replace('addDone', 'readyA').replace('addBranch', 'renamedKernel')
        graph, metadata = self.analyze_text(source)
        self.assertEqual((len(graph['nodes']), len(graph['edges'])), (12, 13))
        self.assertEqual({n['vol'] for n in graph['nodes']}, {513 * 4})
        branches = [n for n in metadata if n['stage'].endswith(':write')]
        first, second, join = branches
        self.assertEqual(first['stream'], 'workerA')
        self.assertNotIn(first['operation_id'], second['ordered_after_operations'])
        self.assertTrue({first['operation_id'], second['operation_id']} <= set(join['ordered_after_operations']))
        parents = {n['id']: set() for n in graph['nodes']}
        for e in graph['edges']: parents[e['target']].add(e['source'])
        def ancestors(i):
            return parents[i] | {a for p in parents[i] for a in ancestors(p)}
        self.assertNotIn(first['id'], ancestors(second['id']))
        self.assertTrue({first['id'], second['id']} <= ancestors(join['id']))

    def test_missing_synchronization_rejected(self):
        source = (ROOT / 'workloads/fork_join_streams.cu').read_text()
        for statement in ['cudaDeviceSynchronize()', 'cudaStreamWaitEvent(joinStream, addDone, 0)',
                          'cudaStreamWaitEvent(joinStream, subtractDone, 0)', 'cudaStreamSynchronize(joinStream)']:
            with self.subTest(statement=statement):
                changed = source.replace('CUDA_CHECK(' + statement + ');', '')
                self.assertNotEqual(source, changed)
                with self.assertRaisesRegex(UnsupportedSource, 'Missing stream/event synchronization'):
                    self.analyze_text(changed)

    def test_event_snapshot_and_wrapper_rejected(self):
        source = (ROOT / 'workloads/fork_join_streams.cu').read_text()
        record = 'CUDA_CHECK(cudaEventRecord(addDone, addStream));'
        changed = source.replace(record, '').replace('addBranch<<<', record + '\n    addBranch<<<')
        with self.assertRaisesRegex(UnsupportedSource, 'Missing stream/event synchronization'):
            self.analyze_text(changed)
        changed = source.replace('std::exit(EXIT_FAILURE);', 'cudaDeviceSynchronize(); std::exit(EXIT_FAILURE);')
        with self.assertRaisesRegex(UnsupportedSource, 'non-diagnostic calls'):
            self.analyze_text(changed)

if __name__ == '__main__': unittest.main()
