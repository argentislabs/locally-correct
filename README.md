# Locally Correct, Globally Wrong

Companion code for Argentis Labs **Working Paper 01** (2026):
*Locally Correct, Globally Wrong: A state-surface partition for authorization
verification in autonomous systems.*

[Paper PDF](papers/paper_01_locally_correct_globally_wrong.pdf) · [Argentis Labs](https://argentislabs.io/research/locally-correct)

---

## The claim

Verification systems for autonomous transactions today evaluate semantic
alignment between a user's stated intent and the proposed transaction
artifact. This evaluation surface is incomplete in a categorical way.
Institutional authorization state — designation chains, disclosed conflicts,
oversight independence — frequently lives outside the transaction artifact and
requires relational reasoning across distributed schema fields.

On the central record (`smoke_002`), frontier LLM-as-judge configurations with
full approver context approve the institutionally invalid record in **five of
five reruns at 0.93–0.97 confidence**. The model reasoning explicitly cites
every relevant field individually and validates each in isolation. A composed
configuration that adds a deterministic predicate-dispatch layer over typed
structural state recovers correct family attribution in five of five reruns
at **sub-millisecond authority-layer latency**.

The architectural conclusion: institutional legitimacy must be represented as
explicit relational state before it can be reliably enforced.

## Reproduce the central result

```bash
git clone https://github.com/argentislabs/locally-correct.git
cd locally-correct
uv sync
```

Run the deterministic authority layer against the harder smoke record:

```bash
uv run python -m eval.harness \
    eval/records/handcrafted/type_d_captured_economic_dependence.json \
    --config authority_partition \
    --n-runs 5
```

Expected output: 5/5 `REJECT(authority_partition.captured_approver)` at
1.0 confidence, ≤1 ms median latency.

To reproduce the full §3.3 results table, run the corresponding configurations
across both records (`type_d_smoke.json` and
`type_d_captured_economic_dependence.json`). LLM-based configurations require
`OPENAI_API_KEY` in the environment.

## Verify every cited number

The paper's results table is auditable, byte-for-byte, against the trace files
in `eval/results/`. The verifier reads `docs/paper_01_citation_log.json` and
checks every assertion against disk:

```bash
uv run python -m eval.verify_citations docs/paper_01_citation_log.json
```

Expected: `43 assertions verified` and exit code 0.

## Repository layout

```
src/authority_partition/        the seven authority predicates + dispatch table
eval/configs/                   the four evaluation configurations (§2 of the paper)
eval/records/handcrafted/       the two smoke records cited in the paper
eval/results/                   per-run trace files (input to citation verifier)
eval/harness.py                 evaluation runner with --n-runs aggregate stats
eval/sweep_deterministic.py     run all seeds through the authority layer
eval/verify_citations.py        citation log verifier (84 assertions)
docs/paper_01_citation_log.json machine-checkable claim log
papers/                         working paper PDF
```

## What's in this repo

- The seven authority predicates and dispatch table (`src/authority_partition/`)
- The four evaluation configurations from §2 of the paper:
  `gpt_no_approver`, `gpt_with_approver`, `authority_partition`, `composed`
- The two handcrafted smoke records the paper cites (`smoke_001`, `smoke_002`)
- The full evaluation harness, results traces, and citation verifier
- The working paper PDF

## What isn't (yet)

- **The cleanliness gate utility.** Used to certify that handcrafted records
  contain no transaction-local alignment defects, so authority-surface failures
  cannot be confounded with local-surface noise. To be released alongside
  Working Paper 03 (cross-family evaluation across all seven authority families).
- **The handcrafted seed corpus that the gate certifies.** Same release moment.
- **A Rust port of the predicate dispatch layer.** Production-grade substrate;
  separate roadmap.

These are deliberate cadence choices. The paper's central claim — that frontier
LLM-as-judge fails on the institutional-relational surface where deterministic
predicate dispatch succeeds — stands fully on what's in this repo.

## Citation

If you use this code or build on the paper:

```
@techreport{argentis2026locally,
  author      = {{Argentis Labs Research}},
  title       = {Locally Correct, Globally Wrong: A state-surface partition
                 for authorization verification in autonomous systems},
  institution = {Argentis Labs},
  type        = {Working Paper},
  number      = {01},
  year        = {2026},
  url         = {https://argentislabs.io/research/locally-correct}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contact

Research and consulting inquiries: [argentislabs.io](https://argentislabs.io)
