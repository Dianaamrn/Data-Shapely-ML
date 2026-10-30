import tempfile
import unittest
from pathlib import Path
from data_shapley.provenance import source_sha256, raw_sha256


class ProvenanceTest(unittest.TestCase):
    def test_line_endings_portable_but_content_and_raw_bytes_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, c = [Path(directory)/name for name in ('lf.py','crlf.py','mixed.py')]
            a.write_bytes(b'x = 1\ny = 2\n')
            b.write_bytes(b'x = 1\r\ny = 2\r\n')
            c.write_bytes(b'x = 1\r\ny = 2\n')
            self.assertEqual(source_sha256(a),source_sha256(b))
            self.assertEqual(source_sha256(a),source_sha256(c))
            self.assertNotEqual(raw_sha256(a),raw_sha256(b))
            c.write_bytes(b'x = 2\ny = 2\n')
            self.assertNotEqual(source_sha256(a),source_sha256(c))
