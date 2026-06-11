from __future__ import annotations

import pandas as pd

from analysis.build_return_y_neutralization_exposures import (
    _build_static_metadata,
    _infer_board,
)


def test_infer_board_maps_szse_chinext_market_type_to_canonical_code() -> None:
    assert _infer_board("302132", "创业板", "SZSE") == "CHINEXT"


def test_infer_board_uses_canonical_codes_for_known_prefixes() -> None:
    assert _infer_board("300750", "创业板", "SZSE") == "CHINEXT"
    assert _infer_board("688981", "科创板", "SSE") == "STAR"
    assert _infer_board("920091", "北交所", "BSE") == "BSE"
    assert _infer_board("600519", "主板", "SSE") == "SSE_MAIN"
    assert _infer_board("000001", "主板", "SZSE") == "SZSE_MAIN"


def test_build_static_metadata_prefers_listing_board_reference(tmp_path) -> None:
    stock_list_path = tmp_path / "stock_list.csv"
    stock_list = pd.DataFrame(
        [
            {
                "TS代码": "302132.SZ",
                "股票代码": "302132",
                "所属行业": "军工电子",
                "市场类型": "创业板",
                "交易所代码": "SZSE",
            },
            {
                "TS代码": "999999.SZ",
                "股票代码": "999999",
                "所属行业": "测试行业",
                "市场类型": "测试市场",
                "交易所代码": "SZSE",
            },
        ]
    )
    stock_list.to_csv(stock_list_path, index=False, encoding="utf-8-sig")

    industry_board_path = tmp_path / "industry_board.txt"
    industry_board_path.write_text(
        "801000\t军工电子\t302132\t测试股票\n"
        "801001\t测试行业\t999999\t测试股票二\n",
        encoding="gbk",
    )

    listing_board_reference_path = tmp_path / "current_snapshot.csv"
    listing_board_reference = pd.DataFrame(
        [
            {
                "ts_code": "302132.SZ",
                "stock_code": "302132",
                "listing_board_code": "CHINEXT",
                "listing_board": "创业板",
            },
            {
                "ts_code": "999999.SZ",
                "stock_code": "999999",
                "listing_board_code": "CHINEXT",
                "listing_board": "创业板",
            },
        ]
    )
    listing_board_reference.to_csv(
        listing_board_reference_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = _build_static_metadata(
        stock_list_path,
        industry_board_path,
        ["302132", "999999"],
        listing_board_reference_path=listing_board_reference_path,
    ).set_index("stock_code")

    assert metadata.loc["302132", "board"] == "CHINEXT"
    assert metadata.loc["999999", "board"] == "CHINEXT"
    assert "SZSE_创业板" not in set(metadata["board"])


def test_build_static_metadata_maps_reference_main_by_exchange(tmp_path) -> None:
    stock_list_path = tmp_path / "stock_list.csv"
    pd.DataFrame(
        [
            {
                "TS代码": "600519.SH",
                "股票代码": "600519",
                "所属行业": "白酒",
                "市场类型": "主板",
                "交易所代码": "SSE",
            },
            {
                "TS代码": "000001.SZ",
                "股票代码": "000001",
                "所属行业": "银行",
                "市场类型": "主板",
                "交易所代码": "SZSE",
            },
        ]
    ).to_csv(stock_list_path, index=False, encoding="utf-8-sig")

    industry_board_path = tmp_path / "industry_board.txt"
    industry_board_path.write_text(
        "801010\t白酒\t600519\t贵州茅台\n"
        "801020\t银行\t000001\t平安银行\n",
        encoding="gbk",
    )

    listing_board_reference_path = tmp_path / "current_snapshot.csv"
    pd.DataFrame(
        [
            {
                "stock_code": "600519.SH",
                "listing_board_code": "MAIN",
                "listing_board": "主板",
                "exchange_code": "SSE",
            },
            {
                "stock_code": "000001.SZ",
                "listing_board_code": "MAIN",
                "listing_board": "主板",
                "exchange_code": "SZSE",
            },
        ]
    ).to_csv(listing_board_reference_path, index=False, encoding="utf-8-sig")

    metadata = _build_static_metadata(
        stock_list_path,
        industry_board_path,
        ["600519", "000001"],
        listing_board_reference_path=listing_board_reference_path,
    ).set_index("stock_code")

    assert metadata.loc["600519", "board"] == "SSE_MAIN"
    assert metadata.loc["000001", "board"] == "SZSE_MAIN"
