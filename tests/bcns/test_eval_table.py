import csv

import torch

from bcns.eval_table import (
    evaluate_inpainting_result,
    summarize_by_method,
    write_summary_csv,
    write_summary_markdown,
)


def _inputs():
    label = torch.zeros(1, 3, 16, 16)
    raw = label.clone()
    raw[..., 6:10, 6:10] = 0.2
    mask = torch.ones(1, 1, 16, 16)
    mask[..., 6:10, 6:10] = 0.0
    measurement = mask * label
    composite = mask * measurement + (1.0 - mask) * raw
    return raw, composite, label, measurement, mask


def test_evaluate_inpainting_result_returns_expected_keys():
    row = evaluate_inpainting_result(*_inputs(), diagnostics={"flow_runtime_sec": 0.5})

    expected = {
        "raw_psnr",
        "composite_psnr",
        "raw_hole_mse",
        "composite_hole_mse",
        "composite_seam_mse",
        "composite_gradient_mismatch",
        "composite_isophote_error",
        "composite_laplacian_mismatch",
        "flow_runtime_sec",
    }
    assert expected.issubset(row.keys())


def test_summarize_by_method_computes_mean_and_std():
    rows = [
        {"method": "a", "composite_psnr": 1.0, "hole_mse": 2.0},
        {"method": "a", "composite_psnr": 3.0, "hole_mse": 4.0},
        {"method": "b", "composite_psnr": 10.0, "hole_mse": ""},
    ]

    summary = summarize_by_method(rows)
    by_method = {row["method"]: row for row in summary}

    assert by_method["a"]["count"] == 2
    assert by_method["a"]["composite_psnr_mean"] == 2.0
    assert by_method["a"]["composite_psnr_std"] == 2 ** 0.5
    assert by_method["b"]["composite_psnr_std"] == 0.0


def test_csv_and_markdown_write_without_error(tmp_path):
    rows = [
        {"method": "a", "composite_psnr": 1.0},
        {"method": "b", "composite_psnr": 2.0, "extra": 3.0},
    ]
    csv_path = tmp_path / "metrics.csv"
    md_path = tmp_path / "summary.md"

    write_summary_csv(rows, csv_path)
    summary = summarize_by_method(rows)
    write_summary_markdown(summary, md_path)

    with csv_path.open() as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == 2
    assert md_path.read_text().startswith("| method | count |")
