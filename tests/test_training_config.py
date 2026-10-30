import dataclasses
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_3050 import update_config

@dataclasses.dataclass
class OptionalRanges:
    range: tuple | None = (0.8,1.2)
    hidden: tuple = (512,256)

class ConfigTests(unittest.TestCase):
    def test_optional_tuple_can_be_disabled(self):
        cfg=OptionalRanges();update_config(cfg,{'range':None,'hidden':[2048,1024]})
        self.assertIsNone(cfg.range)
        self.assertEqual(cfg.hidden,(2048,1024))
    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):update_config(OptionalRanges(),{'typo':1})

if __name__=='__main__':unittest.main()
