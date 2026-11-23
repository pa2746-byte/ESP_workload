"""Host-language parsing needed by complete multi-file applications."""
import tempfile
import unittest
from pathlib import Path
from cuda_configured_model import parse_host_translation
from cuda_source_model import UnsupportedSource, walk

class HostTranslationTest(unittest.TestCase):
    def test_c_and_cpp_resolve_configured_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / 'headers'
            include.mkdir()
            (include / 'value.h').write_text('#define VALUE 7\n')
            for suffix, body in [('.c', 'int main(void) { return VALUE; }'),
                                 ('.cpp', 'class Value { public: int get() { return VALUE; } }; int main() { Value v; return v.get(); }')]:
                source = root / ('main' + suffix)
                source.write_text('#include <value.h>\n' + body)
                ast = parse_host_translation(source, include_dirs=[include])
                self.assertTrue(any(n.get('kind') == 'FunctionDecl' and n.get('name') == 'main' for n in walk(ast)))
            with self.assertRaisesRegex(UnsupportedSource, 'include directory'):
                parse_host_translation(source, include_dirs=[root / 'missing'])

if __name__ == '__main__':
    unittest.main()
