# latent-space-ai-agents-system

複数 LLM 間の **レイテント空間通信**（hidden states / KV cache の直接交換）を検証する AI エージェント。

要件定義書: `REQ-MLLM-AGENT-2026-002 v2.0`（2026-06-25）

## ステータス

| Phase | 状態 | Issue |
|---|---|---|
| 1 基盤キャプチャ | 🟢 実装中 | [#2](https://github.com/sotanengel/latent-space-ai-agents-system/issues/2) |
| 2 再整列・互換性 | ⚪ 雛形のみ | [#3](https://github.com/sotanengel/latent-space-ai-agents-system/issues/3) |
| 3 意味検証 | ⚪ 雛形のみ | [#4](https://github.com/sotanengel/latent-space-ai-agents-system/issues/4) |
| 4 協調・プロバナンス | ⚪ 雛形のみ | [#5](https://github.com/sotanengel/latent-space-ai-agents-system/issues/5) |
| 5 パフォーマンス | ⚪ 雛形のみ | [#6](https://github.com/sotanengel/latent-space-ai-agents-system/issues/6) |
| 6 セキュリティ | ⚪ 雛形のみ | [#7](https://github.com/sotanengel/latent-space-ai-agents-system/issues/7) |
| 7 オンライン運用 | ⚪ 雛形のみ | [#8](https://github.com/sotanengel/latent-space-ai-agents-system/issues/8) |
| 8 マルチモーダル | ⚪ 雛形のみ | [#9](https://github.com/sotanengel/latent-space-ai-agents-system/issues/9) |

マスター: [#1](https://github.com/sotanengel/latent-space-ai-agents-system/issues/1)

## なぜ作るのか

レイテント空間通信は 2025 年以降に登場した新パラダイムで、専用の検証ツールチェーンが存在しない。本リポジトリはその空白を埋める。テキストベース通信（MCP / A2A / LDP）の検証は既存ツール（Pact / Postman / Schemathesis）で十分であり対象外。

## インストール

```bash
git clone https://github.com/sotanengel/latent-space-ai-agents-system.git
cd latent-space-ai-agents-system
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### Takumi Guard（サプライチェーン保護）

CLAUDE.md 準拠で `https://pypi.flatt.tech/` 経由のインストールを推奨。トークンを取得し、

```bash
export TAKUMI_GUARD_TOKEN=tg_anon_xxxxxxxxxxxx
cp .uv.toml.sample .uv.toml         # uv 利用時
# または
cp pip.conf.sample ~/.pip/pip.conf  # pip 利用時
```

の上で `pip install -e ".[dev]"` を実行する。導通確認:

```bash
pip install panda-guard-test-malicious   # 403 Forbidden が返れば成功
```

## 使い方（Phase 1）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from latent_agents.capture import HiddenStateTap, KVCacheTap, TensorSerializer
from latent_agents.analysis import (
    ShapeVerifier, DtypeVerifier, NanInfVerifier, NormStabilityVerifier,
)

tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
inputs = tok("hello world", return_tensors="pt")

with HiddenStateTap(model) as h_tap, KVCacheTap(model) as kv_tap:
    outputs = model(**inputs, output_hidden_states=True, use_cache=True)

for snap in h_tap.snapshots:
    print(ShapeVerifier(expected_hidden_dim=model.config.hidden_size).verify(snap))
    print(NanInfVerifier().verify(snap))

ser = TensorSerializer()
path = ser.save(kv_tap.snapshots, "/tmp/kv.safetensors")
restored = ser.load(path)
assert ser.bit_equal(kv_tap.snapshots, restored)   # FR-KV-001
```

## アーキテクチャ

```
src/latent_agents/
  schemas/         §6 データモデル (pydantic)
  verifier/        検証フレームワーク
  capture/         Phase 1 ★実装
  analysis/        Phase 1/2/7
  semantic/        Phase 3
  coordination/    Phase 4
  quality/         Phase 5/6
  ops/             Phase 7
  multimodal/      Phase 8
```

## 開発

```bash
# Red → Green → Refactor (CLAUDE.md TDD 必須)
pytest tests/unit -v               # ユニット
pytest tests/integration -v        # tiny-gpt2 を使う統合テスト
pre-commit run --all-files         # lint / format / mypy
```

## ライセンス

Apache 2.0
