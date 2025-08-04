#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权 Roll Position 利润变化分析工具
用于计算期权 roll 前后的利润变化、新的盈亏平衡点等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import os
from datetime import datetime

# 设置中文字体（可根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class OptionsRollAnalyzer:
    """期权 Roll Position 分析器"""

    def __init__(self):
        self.data = None
        self.results = None

    def load_data(self, file_path: str) -> None:
        """加载期权 roll 数据"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式，请使用 CSV 或 Excel 文件")

            required_columns = [
                '股票代码', '期权类型', '执行日', '执行价',
                '期权价格', '每股均价', '合同数量', '新的期权价格', '新的执行价格', '平仓价格'
            ]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"缺少必需的列: {missing_columns}")

            # 类型转换
            numeric_cols = ['执行价', '期权价格', '每股均价', '合同数量',
                            '新的期权价格', '新的执行价格', '平仓价格']
            for col in numeric_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data['执行日'] = pd.to_datetime(self.data['执行日'], errors='coerce')

            print(f"成功加载数据，共 {len(self.data)} 条记录")
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise

    def calculate_roll_analysis(self, row: pd.Series) -> Dict:
        """计算单笔期权 roll 的利润变化分析"""
        stock_code = row['股票代码']
        option_type = row['期权类型']
        old_strike_price = row['执行价']
        old_option_premium = row['期权价格']
        avg_cost = row['每股均价']
        contracts = row['合同数量']
        new_option_premium = row['新的期权价格']
        new_strike_price = row['新的执行价格']
        close_price = row['平仓价格']

        # 容错：如果没有提供新的值就回退成旧的
        if pd.isna(new_option_premium):
            new_option_premium = old_option_premium
        if pd.isna(new_strike_price):
            new_strike_price = old_strike_price

        total_shares = contracts * 100

        # 旧期权的总期权费收入（credit）
        old_total_premium = old_option_premium * total_shares

        # 新期权的总期权费收入（credit）
        new_total_premium = new_option_premium * total_shares

        # 平仓成本（debit）
        close_cost = close_price * total_shares

        # 旧仓位已实现盈亏 = 旧期权费收入 - 平仓成本
        old_position_pnl = old_total_premium - close_cost  # 可能为正（盈利）或负（亏损）

        # 每股的旧仓 realized profit，用于调整新成本
        old_realized_profit_per_share = old_position_pnl / total_shares

        # 调整后成本价（旧仓利润/亏损作用到新仓）
        adjusted_cost = avg_cost - old_realized_profit_per_share

        # 新期权收入（新的期权费）
        premium_change = new_total_premium  # 你可以考虑另外增加一个字段表示 net premium delta if desired

        # 因行权价变化带来的影响
        strike_price_change = new_strike_price - old_strike_price
        if option_type.lower().replace(" ", "") in ['sellcall']:
            # sell call: 提高执行价对卖方有利
            strike_price_impact = strike_price_change * total_shares
        elif option_type.lower().replace(" ", "") in ['sellput']:
            # sell put: 降低执行价对卖方有利
            strike_price_impact = (-strike_price_change) * total_shares
        else:
            raise ValueError(f"不支持的期权类型: {option_type}")

        # Roll 后总利润变化 = 旧仓位已实现盈亏 + 新期权费收入 + 执行价变化影响
        total_roll_change = old_position_pnl + premium_change + strike_price_impact

        # 旧盈亏平衡价（基准，用于对比）
        if option_type.lower().replace(" ", "") in ['sellcall']:
            old_profit_breakeven = old_strike_price + old_option_premium
            old_loss_breakeven = avg_cost - old_option_premium
        else:
            old_profit_breakeven = old_strike_price - old_option_premium
            old_loss_breakeven = avg_cost + old_option_premium

        # 新盈利/亏损平衡价（统一使用调整后成本）
        if option_type.lower().replace(" ", "") in ['sellcall']:
            # Sell Call: 盈亏对称围绕调整后成本
            new_profit_breakeven = adjusted_cost + new_option_premium
            new_loss_breakeven = adjusted_cost - new_option_premium
        else:
            # Sell Put: 盈亏对称围绕调整后成本（Put 的盈亏方向相反）
            new_profit_breakeven = adjusted_cost - new_option_premium
            new_loss_breakeven = adjusted_cost + new_option_premium

        return {
            '股票代码': stock_code,
            '期权类型': option_type,
            '旧执行价': old_strike_price,
            '新执行价': new_strike_price,
            '旧期权价格': old_option_premium,
            '新期权价格': new_option_premium,
            '平仓价格': close_price,
            '每股均价': avg_cost,
            '合同数量': contracts,
            '旧期权费收入': old_total_premium,
            '新期权费收入': new_total_premium,
            '平仓成本': close_cost,
            '旧仓位盈亏': old_position_pnl,
            '调整后成本价': adjusted_cost,
            '期权费变化': premium_change,
            '执行价变化': strike_price_change,
            '执行价变化影响': strike_price_impact,
            'Roll后利润变化': total_roll_change,
            '旧盈利平衡价': old_profit_breakeven,
            '新盈利平衡价': new_profit_breakeven,
            '旧亏损平衡价': old_loss_breakeven,
            '新亏损平衡价': new_loss_breakeven,
            '持仓总金额': avg_cost * total_shares
        }

    def analyze_all_rolls(self) -> pd.DataFrame:
        """分析所有期权 roll 交易"""
        results = []
        for idx, row in self.data.iterrows():
            try:
                result = self.calculate_roll_analysis(row)
                results.append(result)
            except Exception as e:
                print(f"计算第 {idx+1} 行时出错: {e}")
                continue
        self.results = pd.DataFrame(results)
        return self.results

    def aggregate_roll_by_stock(self) -> pd.DataFrame:
        """按股票汇总 roll 分析"""
        if self.results is None:
            self.analyze_all_rolls()

        stock_summary = []
        # 预计算每只股票的加权原始成本和旧仓 realized profit per share
        for stock_code in self.results['股票代码'].unique():
            stock_data = self.results[self.results['股票代码'] == stock_code]
            total_contracts = stock_data['合同数量'].sum()
            total_shares = total_contracts * 100

            # 汇总头寸金额（用原始每股均价）
            weighted_cost_numer = (stock_data['每股均价'] * stock_data['合同数量'] * 100).sum()
            weighted_avg_cost = weighted_cost_numer / total_shares

            # 旧期权/新期权/平仓/旧仓位盈亏
            old_premium_sum = stock_data['旧期权费收入'].sum()
            new_premium_sum = stock_data['新期权费收入'].sum()
            close_cost_sum = stock_data['平仓成本'].sum()
            old_position_pnl_sum = stock_data['旧仓位盈亏'].sum()
            roll_change_sum = stock_data['Roll后利润变化'].sum()
            strike_impact_sum = stock_data['执行价变化影响'].sum()
            premium_change_sum = stock_data['期权费变化'].sum()

            # 旧 realized profit per share
            old_realized_profit_per_share = old_position_pnl_sum / total_shares
            adjusted_avg_cost = weighted_avg_cost - old_realized_profit_per_share

            # 加权平均执行价
            weighted_old_strike = (stock_data['旧执行价'] * stock_data['合同数量']).sum() / stock_data['合同数量'].sum()
            weighted_new_strike = (stock_data['新执行价'] * stock_data['合同数量']).sum() / stock_data['合同数量'].sum()

            # 主要类型（用数量比较）
            call_contracts = stock_data[stock_data['期权类型'].str.contains('call', case=False)]['合同数量'].sum()
            put_contracts = stock_data[stock_data['期权类型'].str.contains('put', case=False)]['合同数量'].sum()

            if call_contracts >= put_contracts:
                # sell call
                old_profit_breakeven = weighted_old_strike + (stock_data['旧期权价格'] * stock_data['合同数量']).sum() / total_contracts
                old_loss_breakeven = weighted_avg_cost - (stock_data['旧期权价格'] * stock_data['合同数量']).sum() / total_contracts
                new_profit_breakeven = weighted_new_strike + (stock_data['新期权价格'] * stock_data['合同数量']).sum() / total_contracts
                new_loss_breakeven = adjusted_avg_cost - (stock_data['新期权价格'] * stock_data['合同数量']).sum() / total_contracts
            else:
                # sell put
                old_profit_breakeven = weighted_old_strike - (stock_data['旧期权价格'] * stock_data['合同数量']).sum() / total_contracts
                old_loss_breakeven = weighted_avg_cost + (stock_data['旧期权价格'] * stock_data['合同数量']).sum() / total_contracts
                new_profit_breakeven = weighted_new_strike - (stock_data['新期权价格'] * stock_data['合同数量']).sum() / total_contracts
                new_loss_breakeven = adjusted_avg_cost + (stock_data['新期权价格'] * stock_data['合同数量']).sum() / total_contracts

            stock_summary.append({
                '股票代码': stock_code,
                '旧期权费总收入': old_premium_sum,
                '新期权费总收入': new_premium_sum,
                '平仓总成本': close_cost_sum,
                '旧仓位总盈亏': old_position_pnl_sum,
                '期权费变化': premium_change_sum,
                '执行价变化影响': strike_impact_sum,
                'Roll后总利润': roll_change_sum,
                '利润变化百分比': (roll_change_sum / (weighted_avg_cost * total_shares)) * 100 if weighted_avg_cost * total_shares != 0 else np.nan,
                '加权平均成本价': weighted_avg_cost,
                '汇总调整后成本价': adjusted_avg_cost,
                '加权平均旧执行价': weighted_old_strike,
                '加权平均新执行价': weighted_new_strike,
                '汇总旧盈利平衡价': old_profit_breakeven,
                '汇总新盈利平衡价': new_profit_breakeven,
                '汇总旧亏损平衡价': old_loss_breakeven,
                '汇总新亏损平衡价': new_loss_breakeven,
                '总合同数量': total_contracts
            })

        return pd.DataFrame(stock_summary)

    def save_results(self, filename: str = None, format: str = 'csv') -> str:
        """保存分析结果，返回输出目录路径"""
        if self.results is None:
            self.analyze_all_rolls()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            filename = f"期权Roll分析结果"

        if format.lower() == 'csv':
            detailed_path = os.path.join(output_dir, f"{filename}_详细.csv")
            detailed_results = self.results.copy()
            for col in detailed_results.select_dtypes("number").columns:
                detailed_results[col] = detailed_results[col].round(2)
            detailed_results.to_csv(detailed_path, index=False, encoding='utf-8-sig')

            summary = self.aggregate_roll_by_stock()
            summary_path = os.path.join(output_dir, f"{filename}_汇总.csv")
            for col in summary.select_dtypes("number").columns:
                summary[col] = summary[col].round(2)
            summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

        elif format.lower() == 'excel':
            excel_path = os.path.join(output_dir, f"{filename}.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                detailed_results = self.results.copy()
                for col in detailed_results.select_dtypes("number").columns:
                    detailed_results[col] = detailed_results[col].round(2)
                detailed_results.to_excel(writer, sheet_name='详细分析', index=False)

                summary = self.aggregate_roll_by_stock()
                for col in summary.select_dtypes("number").columns:
                    summary[col] = summary[col].round(2)
                summary.to_excel(writer, sheet_name='股票汇总', index=False)

        print(f"结果已保存到 {output_dir}/")
        return output_dir
    
    def plot_roll_analysis(self, save_plot: bool = True, output_dir: str = None) -> None:
        """只保留两个图：
        1. Roll 前后期权费收入对比
        2. Roll 前后盈亏平衡价格对比（含旧成本和调整后成本）"""
        if self.results is None:
            self.analyze_all_rolls()

        summary = self.aggregate_roll_by_stock()

        # ---- 图1: Roll 前后期权费收入对比 ----
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(summary))
        width = 0.35

        old_prem = summary['旧期权费总收入']
        new_prem = summary['新期权费总收入']

        bars_old = ax1.bar(x - width / 2, old_prem, width, label='旧期权费收入')
        bars_new = ax1.bar(x + width / 2, new_prem, width, label='新期权费收入')

        ax1.set_xticks(x)
        ax1.set_xticklabels(summary['股票代码'], rotation=45)
        ax1.set_ylabel('期权费收入')
        ax1.set_title('Roll 前后期权费收入对比')
        ax1.legend()
        ax1.grid(alpha=0.3)

        for bar in bars_old:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max(1, h * 0.01), f'{h:,.0f}', ha='center', va='bottom', fontsize=8)
        for bar in bars_new:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max(1, h * 0.01), f'{h:,.0f}', ha='center', va='bottom', fontsize=8)

        # ---- 图2: Roll 前后盈亏平衡价格对比（含旧成本 & 调整后成本） ----
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        width = 0.15
        x = np.arange(len(summary))

        old_profit = summary['汇总旧盈利平衡价']
        new_profit = summary['汇总新盈利平衡价']
        old_loss = summary['汇总旧亏损平衡价']
        new_loss = summary['汇总新亏损平衡价']

        ax2.bar(x - 1.5 * width, old_profit, width, label='旧盈利平衡价', color='darkgreen')
        ax2.bar(x - 0.5 * width, new_profit, width, label='新盈利平衡价', color='lightgreen')
        ax2.bar(x + 0.5 * width, old_loss, width, label='旧亏损平衡价', color='darkred')
        ax2.bar(x + 1.5 * width, new_loss, width, label='新亏损平衡价', color='lightcoral')

        # 成本价点：旧成本 & 调整后成本（新成本）
        ax2.scatter(x, summary['加权平均成本价'], marker='o', label='旧成本价', zorder=5)
        ax2.scatter(x, summary['汇总调整后成本价'], marker='X', label='新成本价 (含已实现盈亏调整)', zorder=5)

        ax2.set_xticks(x)
        ax2.set_xticklabels(summary['股票代码'], rotation=45)
        ax2.set_ylabel('价格')
        ax2.set_title('Roll 前后盈亏平衡价格对比（含成本价）')
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.grid(alpha=0.3)

        # 为成本价加数值标签
        for i in range(len(summary)):
            old_c = summary.loc[i, '加权平均成本价']
            adj_c = summary.loc[i, '汇总调整后成本价']
            ax2.text(i, old_c, f'{old_c:.2f}', ha='center', va='bottom', fontsize=7)
            ax2.text(i, adj_c, f'{adj_c:.2f}', ha='center', va='top', fontsize=7)

        plt.tight_layout()

        if save_plot:
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"output/{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
            # 保存两个图
            fig1_path = os.path.join(output_dir, "期权Roll_期权费收入对比.png")
            fig2_path = os.path.join(output_dir, "期权Roll_盈亏平衡价格对比.png")
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {output_dir}/")
        plt.show()
    
    def print_summary(self) -> None:
        """打印分析摘要，包含旧成本价 & 调整后成本价（新成本）"""
        if self.results is None:
            self.analyze_all_rolls()

        summary = self.aggregate_roll_by_stock()

        print("\n" + "=" * 80)
        print("期权 Roll Position 分析摘要")
        print("=" * 80)
        print(f"\n总计分析 {len(self.results)} 笔期权 roll 交易，涉及 {len(summary)} 只股票")
        print("\n各股票 Roll 分析:")
        print("-" * 80)
        for _, row in summary.iterrows():
            print(f"股票代码: {row['股票代码']}")
            print(f"  旧期权费总收入: {row['旧期权费总收入']:,.2f} 元")
            print(f"  新期权费总收入: {row['新期权费总收入']:,.2f} 元")
            print(f"  平仓总成本: {row['平仓总成本']:,.2f} 元")
            print(f"  旧仓位总盈亏: {row['旧仓位总盈亏']:,.2f} 元")
            print(f"  期权费变化: {row['期权费变化']:,.2f} 元")
            print(f"  执行价变化影响: {row['执行价变化影响']:,.2f} 元")
            print(f"  Roll后总利润: {row['Roll后总利润']:,.2f} 元")
            print(f"  利润变化百分比: {row['利润变化百分比']:.2f}%")
            print(f"  旧成本价（加权平均）: {row['加权平均成本价']:.2f} 元")
            print(f"  新成本价（含已实现旧仓盈亏后的调整后成本）: {row['汇总调整后成本价']:.2f} 元")
            print(f"  旧盈利平衡价: {row['汇总旧盈利平衡价']:.2f} 元")
            print(f"  新盈利平衡价: {row['汇总新盈利平衡价']:.2f} 元")
            print(f"  旧亏损平衡价: {row['汇总旧亏损平衡价']:.2f} 元")
            print(f"  新亏损平衡价: {row['汇总新亏损平衡价']:.2f} 元")
            print()

        # 总体统计
        total_old_premium = summary['旧期权费总收入'].sum()
        total_new_premium = summary['新期权费总收入'].sum()
        total_close_cost = summary['平仓总成本'].sum()
        total_old_pnl = summary['旧仓位总盈亏'].sum()
        total_roll_change = summary['Roll后总利润'].sum()
        total_investment = self.results['持仓总金额'].sum()
        overall_change_percentage = (total_roll_change / total_investment) * 100 if total_investment != 0 else float('nan')

        print("总体统计:")
        print("-" * 40)
        print(f"旧期权费总收入: {total_old_premium:,.2f} 元")
        print(f"新期权费总收入: {total_new_premium:,.2f} 元")
        print(f"平仓总成本: {total_close_cost:,.2f} 元")
        print(f"旧仓位总盈亏: {total_old_pnl:,.2f} 元")
        print(f"Roll后总利润: {total_roll_change:,.2f} 元")
        print(f"Roll后总体利润百分比: {overall_change_percentage:.2f}%")
        print("=" * 80)


def create_sample_roll_data() -> pd.DataFrame:
    """创建示例 roll 数据"""
    sample_data = {
        '股票代码': ['AAPL', 'AAPL', 'TSLA', 'TSLA', 'MSFT'],
        '期权类型': ['sell call', 'sell put', 'sell call', 'sell put', 'sell call'],
        '执行日': ['2024-03-15', '2024-03-15', '2024-04-19', '2024-04-19', '2024-05-17'],
        '执行价': [180, 160, 250, 220, 400],
        '期权价格': [5.0, 3.0, 8.0, 6.0, 12.0],
        '每股均价': [175, 175, 240, 240, 380],
        '合同数量': [1, 1, 2, 1, 1],
        '新的期权价格': [6.5, 2.5, 9.0, 5.5, 13.5],
        '新的执行价格': [185, 155, 260, 215, 410]
    }
    
    return pd.DataFrame(sample_data)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='期权 Roll Position 分析工具')
    parser.add_argument('--input', '-i', help='输入文件路径 (CSV或Excel)')
    parser.add_argument('--output', '-o', help='输出文件名前缀')
    parser.add_argument('--format', '-f', choices=['csv', 'excel'], default='csv', help='输出格式')
    parser.add_argument('--plot', '-p', action='store_true', help='生成图表')
    parser.add_argument('--create-sample', '-s', action='store_true', help='创建示例数据文件')
    
    args = parser.parse_args()
    
    analyzer = OptionsRollAnalyzer()
    
    if args.create_sample:
        # 创建示例数据
        sample_data = create_sample_roll_data()
        sample_data.to_csv('示例期权Roll数据.csv', index=False, encoding='utf-8-sig')
        print("已创建示例数据文件: 示例期权Roll数据.csv")
        print("请编辑此文件，填入你的实际期权 roll 数据")
        return
    
    if args.input:
        # 加载数据
        analyzer.load_data(args.input)
        
        # 执行分析
        analyzer.analyze_all_rolls()
        
        # 打印摘要
        analyzer.print_summary()
        
        # 保存结果
        if args.output:
            output_dir = analyzer.save_results(args.output, args.format)
        else:
            output_dir = analyzer.save_results(format=args.format)
        
        # 生成图表
        if args.plot:
            analyzer.plot_roll_analysis(output_dir=output_dir)
    
    else:
        print("请提供输入文件路径，或使用 --create-sample 创建示例数据文件")
        print("使用示例:")
        print("  python options_roll_analyzer.py --create-sample")
        print("  python options_roll_analyzer.py --input 示例期权Roll数据.csv --plot")


if __name__ == "__main__":
    main() 