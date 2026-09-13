# Dual Translate mapping curriculum

Beginner track for the **2026-09-11** Dual mapping pass. Not CXR Foundations. Not RepEng Ph9–14 (`:8262`).

**Claim:** mapping, not selection. Not auto-correct.

| | |
|--|--|
| Track UI | http://127.0.0.1:8264/ |
| Map viewer | http://127.0.0.1:8263/ |
| PDF | `curriculum-mapping/CXR-Mapping-Curriculum.pdf` (copy also under `notes/`) |
| Lab | this repo (`cxr-evidence-grounding-lab/`) |

```bash
cd ~/staging/cxr-evidence-grounding-lab
./scripts/run_mapping_curriculum.sh          # :8264
./scripts/run_map_viewer.sh                  # :8263
./curriculum-mapping/modules/build-mapping-pdf.sh
../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py I2 C4 T1 T2
```

Start **MAP-00**.
