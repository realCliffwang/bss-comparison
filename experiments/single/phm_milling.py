"""
PHM 2010 铣削刀具磨损实验
用法: python -m experiments.single.phm_milling
"""

from bss_test.utils.logger import setup_logging

from experiments._common import (
    load_phm_signals, run_bss_experiment, write_csv_summary, init_experiment,
)


def process_single_tool(tool_id, config, output_dir):
    """处理单个刀具的数据"""
    from bss_test.io.phm import load_phm_cut, load_phm_wear
    from bss_test.evaluation import plot_wear_evolution

    logger = init_experiment.__wrapped_logger__ if hasattr(init_experiment, '__wrapped_logger__') else None
    from bss_test.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info(f"\n{'='*60}")
    logger.info(f"处理刀具 c{tool_id}")
    logger.info(f"{'='*60}")

    tool_output_dir = output_dir / f"c{tool_id}"
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n[1/3] 加载 c{tool_id} 数据...")
    try:
        signals_pre, fs_pre = load_phm_signals(config, tool_id=tool_id, max_duration=1.0)
        logger.info(f"  Shape: {signals_pre.shape}, fs: {fs_pre} Hz")
    except Exception as e:
        logger.error(f"  加载失败: {e}")
        return None

    logger.info(f"\n[2/3] 运行 BSS + 包络谱分析...")
    result = run_bss_experiment(
        config, signals_pre, fs_pre, tool_output_dir,
        title_prefix=f"PHM 2010 c{tool_id} cut150 — ",
    )

    # 磨损演化分析
    if tool_id in [1, 4, 6]:
        logger.info(f"\n  生成 c{tool_id} 磨损演化图...")
        try:
            from bss_test.preprocessing import preprocess_signals
            from bss_test.tfa import build_observation_matrix
            from bss_test.bss import bss_factory
            from experiments._common import build_tfa_config, build_preprocess_config

            wear = load_phm_wear(tool_id=tool_id, data_dir=config.data_dir)
            S_list = []
            wear_labels = []
            n_cuts_to_analyze = min(10, len(wear))

            tfa_config = build_tfa_config(config)

            for cut_idx in range(1, n_cuts_to_analyze + 1):
                try:
                    signals_cut, fs_cut, _ = load_phm_cut(
                        tool_id=tool_id, cut_no=cut_idx,
                        data_dir=config.data_dir,
                        sensor_types=["vib_x", "vib_y", "vib_z"],
                    )
                    n_use_cut = min(signals_cut.shape[1], int(0.5 * fs_cut))
                    signals_cut = signals_cut[:, :n_use_cut]
                    signals_pre_cut, fs_pre_cut = preprocess_signals(
                        signals_cut, fs_cut, build_preprocess_config(config)
                    )
                    X_cut, _ = build_observation_matrix(signals_pre_cut, fs_pre_cut, tfa_config)
                    S_est_cut, _, _ = bss_factory(
                        X_cut, method=config.bss.method, n_components=config.bss.n_sources
                    )
                    S_list.append(S_est_cut)
                    wear_labels.append(wear[cut_idx - 1])
                except Exception as e:
                    logger.warning(f"    加载 cut {cut_idx} 失败: {e}")
                    continue

            if S_list:
                import matplotlib.pyplot as plt
                fig, _ = plot_wear_evolution(
                    S_list, wear_labels, tool_id=tool_id,
                    title_prefix=f"PHM 2010 c{tool_id} — ",
                )
                fig.savefig(tool_output_dir / "wear_evolution.png", dpi=200)
                plt.close(fig)
                logger.info(f"    磨损演化图生成成功")
            else:
                logger.warning(f"    没有有效的切割数据用于磨损演化分析")
        except Exception as e:
            logger.warning(f"  磨损演化图生成失败: {e}")

    return {
        "tool_id": tool_id,
        "independence": result["independence"],
        "ffds": result["ffds"],
        "n_sources": result["n_sources"],
    }


def main():
    """运行 PHM 2010 实验"""
    from bss_test.utils.logger import get_logger
    logger = get_logger(__name__)

    config, output_dir, _ = init_experiment(
        "PHM 2010 铣削刀具磨损实验", "phm2010", "configs/phm2010.yaml",
    )
    logger.info("处理所有刀具: c1, c2, c3, c4, c5, c6")

    all_results = []
    for tool_id in range(1, 7):
        result = process_single_tool(tool_id, config, output_dir)
        if result:
            all_results.append(result)

    logger.info("\n" + "=" * 60)
    logger.info("汇总报告")
    logger.info("=" * 60)
    logger.info(f"{'刀具':<8} {'独立性':<12} {'FFDS':<10} {'源数量':<8}")
    logger.info("-" * 38)
    for result in all_results:
        logger.info(f"c{result['tool_id']:<7} {result['independence']:<12.4f} {result['ffds']:<10.2f} {result['n_sources']:<8}")

    write_csv_summary(output_dir, all_results)
    logger.info(f"\n完成! 结果保存到: {output_dir}")


if __name__ == "__main__":
    setup_logging(level="info")
    main()
