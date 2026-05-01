"""
PHM 2010 铣削刀具磨损实验
用法: python -m experiments.single.phm_milling
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bss_test import (
    preprocess_signals,
    ExperimentConfig,
)
from bss_test.io.phm import load_phm_cut, load_phm_wear
from bss_test.tfa import build_observation_matrix, cwt_transform
from bss_test.bss import bss_factory
from bss_test.evaluation import (
    compute_independence_metric,
    compute_fault_detection_score,
    plot_envelope_spectrum,
    plot_correlation_matrix,
    plot_wear_evolution,
    setup_academic_style,
)
from bss_test.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def process_single_tool(tool_id, config, output_dir):
    """处理单个刀具的数据"""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理刀具 c{tool_id}")
    logger.info(f"{'='*60}")
    
    # 创建刀具输出目录
    tool_output_dir = output_dir / f"c{tool_id}"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info(f"\n[1/5] 加载 c{tool_id} 数据...")
    try:
        signals, fs, cut_no = load_phm_cut(
            tool_id=tool_id,
            cut_no=150,
            data_dir=config.data_dir,
            sensor_types=["vib_x", "vib_y", "vib_z"],
        )
        logger.info(f"  加载 {signals.shape[0]} 通道, {signals.shape[1]} 样本 @ {fs} Hz")
    except Exception as e:
        logger.error(f"  加载失败: {e}")
        return None
    
    # 使用子集进行快速处理
    n_use = min(signals.shape[1], int(1.0 * fs))
    signals = signals[:, :n_use]
    logger.info(f"  使用前 {n_use} 样本 ({n_use/fs:.2f} 秒)")
    
    # 预处理
    logger.info("\n[2/5] 预处理...")
    preprocess_config = {
        "detrend": config.preprocess.detrend,
        "bandpass": config.preprocess.bandpass,
        "normalize": config.preprocess.normalize,
    }
    signals_pre, fs_pre = preprocess_signals(signals, fs, preprocess_config)
    logger.info(f"  预处理后形状: {signals_pre.shape}, fs: {fs_pre} Hz")
    
    # 构建观测矩阵
    logger.info("\n[3/5] 构建观测矩阵...")
    cwt_config = {
        "mode": config.cwt.mode,
        "tfa_method": config.cwt.tfa_method,
        "n_bands": config.cwt.n_bands,
        "freq_range": config.cwt.freq_range,
        "wavelet": config.cwt.wavelet,
        "bands_per_ch": config.cwt.bands_per_ch,
    }
    X, labels = build_observation_matrix(signals_pre, fs_pre, cwt_config)
    logger.info(f"  观测矩阵: {X.shape[0]} 观测 × {X.shape[1]} 样本")
    
    # BSS 分离
    logger.info(f"\n[4/5] 运行 BSS ({config.bss.method})...")
    S_est, A_est, W = bss_factory(
        X, 
        method=config.bss.method, 
        n_components=config.bss.n_sources
    )
    logger.info(f"  估计源: {S_est.shape[0]} × {S_est.shape[1]}")
    
    # 评估
    logger.info("\n[5/5] 评估...")
    indep = compute_independence_metric(S_est)
    ffds = compute_fault_detection_score(S_est, fs_pre, config.feature_freqs)
    
    logger.info(f"  独立性指标: {indep:.4f}")
    logger.info(f"  故障频率检测分数: {ffds:.2f}")
    
    # 可视化
    fig, _ = plot_envelope_spectrum(
        S_est, fs_pre,
        fault_freqs=config.feature_freqs,
        title_prefix=f"PHM 2010 c{tool_id} cut150 — "
    )
    fig.savefig(tool_output_dir / "envelope_spectrum.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_correlation_matrix(S_est, title=f"PHM 2010 c{tool_id} Source Correlation")
    fig.savefig(tool_output_dir / "correlation_matrix.png", dpi=200)
    plt.close(fig)
    
    # 磨损演化分析（仅对训练数据 c1, c4, c6）
    if tool_id in [1, 4, 6]:
        logger.info(f"\n  生成 c{tool_id} 磨损演化图...")
        try:
            # 加载磨损数据
            wear = load_phm_wear(tool_id=tool_id, data_dir=config.data_dir)
            
            # 加载多个切割的数据用于磨损演化分析
            S_list = []
            wear_labels = []
            n_cuts_to_analyze = min(10, len(wear))  # 分析前10个切割
            
            for cut_idx in range(1, n_cuts_to_analyze + 1):
                try:
                    signals_cut, fs_cut, _ = load_phm_cut(
                        tool_id=tool_id,
                        cut_no=cut_idx,
                        data_dir=config.data_dir,
                        sensor_types=["vib_x", "vib_y", "vib_z"],
                    )
                    # 使用子集
                    n_use_cut = min(signals_cut.shape[1], int(0.5 * fs_cut))
                    signals_cut = signals_cut[:, :n_use_cut]
                    
                    # 预处理
                    signals_pre_cut, fs_pre_cut = preprocess_signals(signals_cut, fs_cut, preprocess_config)
                    
                    # 构建观测矩阵
                    X_cut, _ = build_observation_matrix(signals_pre_cut, fs_pre_cut, cwt_config)
                    
                    # BSS 分离
                    S_est_cut, _, _ = bss_factory(
                        X_cut, 
                        method=config.bss.method, 
                        n_components=config.bss.n_sources
                    )
                    
                    S_list.append(S_est_cut)
                    wear_labels.append(wear[cut_idx - 1])
                except Exception as e:
                    logger.warning(f"    加载 cut {cut_idx} 失败: {e}")
                    continue
            
            if S_list:
                fig, _ = plot_wear_evolution(
                    S_list, 
                    wear_labels,
                    tool_id=tool_id,
                    title_prefix=f"PHM 2010 c{tool_id} — "
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
        "independence": indep,
        "ffds": ffds,
        "n_sources": S_est.shape[0],
    }


def main():
    """运行 PHM 2010 实验"""
    setup_academic_style()
    # 加载配置
    config = ExperimentConfig.from_yaml("configs/phm2010.yaml")
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHM 2010 铣削刀具磨损实验")
    logger.info("=" * 60)
    logger.info("处理所有刀具: c1, c2, c3, c4, c5, c6")
    
    # 处理所有刀具
    all_results = []
    for tool_id in range(1, 7):
        result = process_single_tool(tool_id, config, output_dir)
        if result:
            all_results.append(result)
    
    # 生成汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("汇总报告")
    logger.info("=" * 60)
    logger.info(f"{'刀具':<8} {'独立性':<12} {'FFDS':<10} {'源数量':<8}")
    logger.info("-" * 38)
    for result in all_results:
        logger.info(f"c{result['tool_id']:<7} {result['independence']:<12.4f} {result['ffds']:<10.2f} {result['n_sources']:<8}")
    
    # 保存汇总 CSV
    import csv
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool_id", "independence", "ffds", "n_sources"])
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"\n汇总报告保存到: {csv_path}")
    
    logger.info(f"\n完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging(level="info")
    main()
