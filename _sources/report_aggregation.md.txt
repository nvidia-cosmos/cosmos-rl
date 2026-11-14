# Report aggregation

`cosmos_rl.utils.util.aggregate_report_data(reports, destination, prefix="")`
projects producer reports into numeric metrics and media for logging. It updates
and returns the caller's destination dictionary. Existing destination entries
take precedence, including when a prefix is supplied.

Numeric observations are finite Python `int`, `float`, and `bool` values and
NumPy integer, floating, and boolean scalars. `None`, NaN, infinity, strings,
complex numbers, lists, dictionaries, NumPy arrays (including zero-dimensional
arrays), and tensors (including single-element tensors) are excluded. Aggregation
does not call conversion, stringification, or device-copy methods on excluded
values. Producers should explicitly convert intended tensor metrics into host
scalars before submitting a report.

Each key is reduced over its supported observations. Keys with no supported
observations are omitted. The original reports and all their values remain
unchanged, so application consumers can still read structured diagnostics or
per-step lists directly from those reports.

For compatibility, a report that **omits** a numeric key contributes zero when
that key has at least one supported observation. An explicitly invalid value
is skipped. For example:

| Reports | Result |
| --- | --- |
| `[{"score": 2.0}, {"score": 4.0}]` | `{"score": 3.0}` |
| `[{"score": 2.0}, {}]` | `{"score": 1.0}` |
| `[{"score": 2.0}, {"score": None}]` | `{"score": 2.0}` |
| `[{"score": 2.0}, {"score": NaN}, {}]` | `{"score": 1.0}` |
| `[{"score": None}, {}]` | `{}` |

Suffix reducers retain their existing definitions: `_max` and `_min` take the
extrema; `_sum`, `_count`, and `_cnt` sum; `_std` takes the root mean square of
reported standard deviations; every other suffix uses the mean. The missing-key
zero convention applies to all these reducers, including extrema and `_std`.

`rollout_images` and `rollout_videos` bypass numeric filtering. Each uses the
first report containing that key and preserves the media object's identity,
including when its value is `None`.

Run the CPU regressions with:

```sh
python -m pytest -q tests/test_report_aggregation.py
```
