import unittest

from profile_workload_memory import compare, parse_csv


LONG = '''==PROF== Connected
"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","123","vectorAdd(const float *, const float *, float *, int)","dram__bytes_read.sum","byte","8,388,608"
"0","123","vectorAdd(const float *, const float *, float *, int)","dram__bytes_write.sum","byte","4194304"
'''


class HardwareMemoryTest(unittest.TestCase):
    def test_long_and_matching(self):
        rows = parse_csv('\n\n' + LONG)
        result = compare("vector_add", rows)[0]
        self.assertEqual(result["logical_read_bytes"], 8388608)
        self.assertEqual(result["measurement"]["metrics"]["dram_write_bytes"], 4194304)

    def test_wide_with_units(self):
        rows = parse_csv('''"ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum"
"","","byte","byte"
"0","vectorAdd","128","64"
''')
        self.assertEqual(rows[0]["metrics"]["dram_read_bytes"], 128)

    def test_reject_unavailable_scaled_and_duplicates(self):
        for invalid in [LONG.replace('"4194304"', '"n/a"'),
                        LONG.replace('"byte"', '"Kbyte"'),
                        LONG + LONG.splitlines()[-1] + '\n',
                        LONG.replace('"4194304"', '"NaN"'),
                        '\n'.join(LONG.splitlines()[:-1])]:
            with self.assertRaises(ValueError):
                parse_csv(invalid)

    def test_missing_or_repeated_kernel_rejected(self):
        records = parse_csv(LONG)
        with self.assertRaises(ValueError):
            compare("fork_join", records)
        with self.assertRaises(ValueError):
            compare("vector_add", records + records)


if __name__ == "__main__":
    unittest.main()
