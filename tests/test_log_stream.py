import io
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
import backend_bridge as bridge


class ConsoleStreamTest(unittest.TestCase):
    def test_cr_progress_and_crlf_are_separate_records(self):
        source = io.BytesIO("  1/8  loss=0.5\r  2/8  loss=0.4\r\n完成\n".encode("utf-8"))
        sink = io.BytesIO()
        records = list(bridge.iter_console_records(source, sink))
        decoded = [bridge.decode_console_line(record)[0] for record in records]
        self.assertEqual(decoded, ["  1/8  loss=0.5", "  2/8  loss=0.4", "完成"])
        self.assertEqual(sink.getvalue(), source.getvalue())

    def test_utf8_and_gb18030_records_decode_without_replacement(self):
        utf8_text, utf8_encoding = bridge.decode_console_line("加载模型".encode("utf-8"))
        gbk_text, gbk_encoding = bridge.decode_console_line("缓存完成".encode("gb18030"))
        self.assertEqual(utf8_text, "加载模型")
        self.assertEqual(gbk_text, "缓存完成")
        self.assertNotIn("replace", utf8_encoding)
        self.assertNotIn("replace", gbk_encoding)

    def test_optional_deepspeed_probe_is_noise(self):
        self.assertEqual(bridge.record_kind("LINK : warning LNK1181: cannot open input file 'aio.lib'"), "noise")
        self.assertEqual(bridge.record_kind("LINK : warning LNK1181: cannot open input file 'cufile.lib'"), "noise")
        self.assertEqual(bridge.record_kind("RuntimeError: CUDA out of memory"), "error")


if __name__ == "__main__":
    unittest.main()
