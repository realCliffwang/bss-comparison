"""
NASA 铣削数据集实验
用法: python -m experiments.single.nasa_milling
"""

from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_nasa_signals, run_bss_experiment, write_csv_summary, init_experiment,
)


def main():
    """运行 NASA 铣削实验"""
    config, output_dir, logger = init_experiment(
        "NASA 铣削数据集实验", "nasa_milling", "configs/nasa.yaml",
    )

    logger.info("\n[1/3] 加载 NASA 数据...")
    signals_pre, fs_pre, meta = load_nasa_signals(config, run_index=0, max_duration=5.0)
    logger.info(f"  Shape: {signals_pre.shape}, fs: {fs_pre} Hz")
    logger.info(f"  Case: {meta['case']}, Run: {meta['run']}, VB: {meta['VB']}")

    logger.info("\n[2/3] 运行 BSS + 包络谱分析...")
    result = run_bss_experiment(
        config, signals_pre, fs_pre, output_dir,
        title_prefix=f"NASA Run {meta['run']} (Case {meta['case']}) — ",
    )

    logger.info("\n[3/3] 完成!")
    write_csv_summary(output_dir, [{"run": meta["run"], "case": meta["case"], **result}])
    logger.info(f"结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
