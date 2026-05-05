"""
CWRU 轴承故障诊断实验
用法: python -m experiments.single.cwru
"""

from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_cwru_signals, run_bss_experiment, write_csv_summary, init_experiment,
)


def main():
    """运行 CWRU 实验"""
    config, output_dir, logger = init_experiment(
        "CWRU 轴承故障诊断实验", "cwru", "configs/cwru.yaml",
    )

    logger.info("\n[1/3] 加载 CWRU 数据...")
    signals_pre, fs_pre = load_cwru_signals(config, fault_type="inner_race_007", max_duration=2.0)
    logger.info(f"  Shape: {signals_pre.shape}, fs: {fs_pre} Hz")

    logger.info("\n[2/3] 运行 BSS + 包络谱分析...")
    result = run_bss_experiment(
        config, signals_pre, fs_pre, output_dir,
        title_prefix="CWRU inner_race_007 — ",
    )

    logger.info("\n[3/3] 完成!")
    write_csv_summary(output_dir, [{"fault": "inner_race_007", **result}])
    logger.info(f"结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
