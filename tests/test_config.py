from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.config import EXTRA_CONFIG_ENV, load_config


class ConfigTests(unittest.TestCase):
    def test_extra_config_env_merges_after_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            extra_path = Path(tmp) / "extra.json"
            extra_path.write_text(
                json.dumps(
                    {
                        "server": {"port": 4099},
                        "modalities": {
                            "image": {"enabled": True},
                            "audio": {"enabled": True},
                        },
                    }
                )
            )

            with patch.dict("os.environ", {EXTRA_CONFIG_ENV: str(extra_path)}, clear=True):
                config = load_config()

        self.assertEqual(config["server"]["port"], 4099)
        self.assertTrue(config["modalities"]["image"]["enabled"])
        self.assertTrue(config["modalities"]["audio"]["enabled"])


if __name__ == "__main__":
    unittest.main()
