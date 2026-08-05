# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from jlens import cauldron


class _Image:
    mode = "RGB"
    size = (1, 1)

    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return bytes([self.value, 0, 0])

    def copy(self):
        return _Image(self.value)


def test_manifest_is_pinned_hashed_and_reloadable(tmp_path, monkeypatch):
    rows = [
        {
            "images": [_Image(index)],
            "texts": [
                {
                    "user": f"question {index}",
                    "assistant": f"answer {index}",
                    "source": "fake",
                }
            ],
        }
        for index in range(100)
    ]

    def load_dataset(repo, config, split, revision, streaming):
        assert repo == cauldron.CAULDRON_REPO
        assert revision == "fixed-sha"
        return rows

    monkeypatch.setattr(cauldron, "_require_datasets", lambda: load_dataset)
    monkeypatch.setattr(
        cauldron, "_resolved_revision", lambda repo, revision: "fixed-sha"
    )
    path = tmp_path / "manifest.jsonl"
    records = cauldron.build_cauldron_manifest(
        path,
        seed=7,
        quotas={"fake": {"fit": 1, "eval": 1}},
    )
    assert {record["split"] for record in records} == {"fit", "eval"}
    assert all(record["revision"] == "fixed-sha" for record in records)
    assert all(
        "image_sha256" in record and "images" not in record for record in records
    )

    samples = list(cauldron.iter_cauldron_samples(path, split="fit"))
    assert len(samples) == 1
    assert samples[0].metadata["revision"] == "fixed-sha"
    assert samples[0].user_text.startswith("question")
