#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权交易分析工具
用于计算期权交易的盈亏平衡点、利润分析等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptionsAnalyzer:
    """期权交易分析器"""
    
    def __init__(self):
        self.data = None
        self.results = None
        
    def load_data(self, file_path: str) -> None:
        """加载期权交易数据"""
        try:
            # 尝试读取CSV文件
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
            
            # 验证必需的列
            required_columns = [
                '股票代码', '期权类型', '执行日', '执行价', 
                '期权价格', '每股均价', '合同数量'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"缺少必需的列: {missing_columns}")
            
            # 数据类型转换
            self.data['执行价'] = pd.to_numeric(self.data['执行价'], errors='coerce')
            self.data['期权价格'] = pd.to_numeric(self.data['期权价格'], errors='coerce')
            self.data['每股均价'] = pd.to_numeric(self.data['每股均价'], errors='coerce')
            self.data['合同数量'] = pd.to_numeric(self.data['合同数量'], errors='coerce')
            self.data['执行日'] = pd.to_datetime(self.data['执行日'], errors='coerce')
            
            print(f"成功加载数据，共 {len(self.data)} 条记录")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def calculate_single_option(self, row: pd.Series) -> Dict:
        """计算单笔期权交易的盈亏分析"""
        stock_code = row['股票代码']
        option_type = row['期权类型']
        strike_price = row['执行价']
        option_premium = row['期权价格']
        avg_cost = row['每股均价']
        contracts = row['合同数量']
        
        # 每股期权费（用户输入的是每股期权费）
        premium_per_share = option_premium  # 用户输入的是每股期权费
        # 每份合同的期权费
        premium_per_contract = option_premium * 100  # 每股期权费 * 100股
        
        # 计算行权利润
        if option_type.lower() in ['sell call', 'sellcall']:
            # Sell Call: 如果被行权，以执行价卖出股票
            # 利润 = 期权费 + (执行价 - 成本价) * 100股
            assignment_profit = premium_per_share + (strike_price - avg_cost)
            assignment_profit_total = assignment_profit * 100 * contracts
            
        elif option_type.lower() in ['sell put', 'sellput']:
            # Sell Put: 如果被行权，以执行价买入股票
            # 利润 = 期权费 - (成本价 - 执行价) * 100股
            assignment_profit = premium_per_share - (avg_cost - strike_price)
            assignment_profit_total = assignment_profit * 100 * contracts
            
        else:
            raise ValueError(f"不支持的期权类型: {option_type}")
        
        # 计算持仓总金额
        total_investment = avg_cost * 100 * contracts
        
        # 利润相对于本金的百分比
        profit_percentage = (assignment_profit_total / total_investment) * 100
        
        # 计算盈亏平衡价格
        if option_type.lower() in ['sell call', 'sellcall']:
            # Sell Call: 股价涨到执行价+期权费时开始无利润
            profit_breakeven = strike_price + premium_per_share
            # 股价跌到成本价-期权费时开始亏损
            loss_breakeven = avg_cost - premium_per_share
        else:
            # Sell Put: 股价跌到执行价-期权费时开始无利润
            profit_breakeven = strike_price - premium_per_share
            # 股价涨到成本价+期权费时开始亏损
            loss_breakeven = avg_cost + premium_per_share
        
        return {
            '股票代码': stock_code,
            '期权类型': option_type,
            '执行价': strike_price,
            '期权价格': option_premium,
            '每股均价': avg_cost,
            '合同数量': contracts,
            '行权利润': assignment_profit_total,
            '利润百分比': profit_percentage,
            '盈利平衡价': profit_breakeven,
            '亏损平衡价': loss_breakeven,
            '持仓总金额': total_investment
        }
    
    def analyze_all_options(self) -> pd.DataFrame:
        """分析所有期权交易"""
        results = []
        
        for idx, row in self.data.iterrows():
            try:
                result = self.calculate_single_option(row)
                results.append(result)
            except Exception as e:
                print(f"计算第 {idx+1} 行时出错: {e}")
                continue
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def aggregate_by_stock(self) -> pd.DataFrame:
        """按股票汇总分析"""
        if self.results is None:
            self.analyze_all_options()
        
        # 按股票代码分组
        grouped = self.results.groupby('股票代码').agg({
            '行权利润': 'sum',
            '持仓总金额': 'sum',
            '合同数量': 'sum'
        }).reset_index()
        
        # 计算加权平均成本价
        weighted_avg_cost = []
        for stock_code in grouped['股票代码']:
            stock_data = self.results[self.results['股票代码'] == stock_code]
            total_value = (stock_data['每股均价'] * stock_data['合同数量'] * 100).sum()
            total_contracts = stock_data['合同数量'].sum()
            weighted_avg = total_value / (total_contracts * 100)
            weighted_avg_cost.append(weighted_avg)
        
        grouped['加权平均成本价'] = weighted_avg_cost
        
        # 计算汇总利润百分比
        grouped['总利润百分比'] = (grouped['行权利润'] / grouped['持仓总金额']) * 100
        
        # 计算汇总盈亏平衡价格（简化计算）
        stock_summary = []
        for stock_code in grouped['股票代码']:
            stock_data = self.results[self.results['股票代码'] == stock_code]
            
            # 计算加权平均期权费（用户输入的是每股期权费）
            total_premium = (stock_data['期权价格'] * stock_data['合同数量']).sum()
            total_contracts = stock_data['合同数量'].sum()
            avg_premium_per_share = total_premium / total_contracts  # 直接计算每股平均期权费
            
            # 计算加权平均执行价
            weighted_strike = (stock_data['执行价'] * stock_data['合同数量']).sum() / stock_data['合同数量'].sum()
            
            # 判断主要期权类型
            call_contracts = stock_data[stock_data['期权类型'].str.contains('call', case=False)]['合同数量'].sum()
            put_contracts = stock_data[stock_data['期权类型'].str.contains('put', case=False)]['合同数量'].sum()
            
            if call_contracts > put_contracts:
                # 主要是Sell Call
                profit_breakeven = weighted_strike + avg_premium_per_share
                loss_breakeven = weighted_avg_cost[stock_summary.__len__()] - avg_premium_per_share
            else:
                # 主要是Sell Put
                profit_breakeven = weighted_strike - avg_premium_per_share
                loss_breakeven = weighted_avg_cost[stock_summary.__len__()] + avg_premium_per_share
            
            stock_summary.append({
                '股票代码': stock_code,
                '总行权利润': stock_data['行权利润'].sum(),
                '总持仓金额': stock_data['持仓总金额'].sum(),
                '总利润百分比': (stock_data['行权利润'].sum() / stock_data['持仓总金额'].sum()) * 100,
                '加权平均成本价': weighted_avg_cost[stock_summary.__len__()],
                '加权平均执行价': weighted_strike,
                '汇总盈利平衡价': profit_breakeven,
                '汇总亏损平衡价': loss_breakeven,
                '总合同数量': stock_data['合同数量'].sum()
            })
        
        return pd.DataFrame(stock_summary)
    
    def save_results(self, filename: str = None, format: str = 'csv') -> str:
        """保存分析结果，返回输出目录路径"""
        if self.results is None:
            self.analyze_all_options()
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"期权分析结果"
        
        if format.lower() == 'csv':
            # 保存详细结果，保留2位小数
            detailed_path = os.path.join(output_dir, f"{filename}_详细.csv")
            # 对数值列进行四舍五入到2位小数
            detailed_results = self.results.copy()
            numeric_columns = ['执行价', '期权价格', '每股均价', '行权利润', '利润百分比', '盈利平衡价', '亏损平衡价', '持仓总金额']
            for col in numeric_columns:
                if col in detailed_results.columns:
                    detailed_results[col] = detailed_results[col].round(2)
            detailed_results.to_csv(detailed_path, index=False, encoding='utf-8-sig')
            
            # 保存汇总结果，保留2位小数
            summary = self.aggregate_by_stock()
            summary_path = os.path.join(output_dir, f"{filename}_汇总.csv")
            # 对汇总结果的数值列进行四舍五入到2位小数
            summary_numeric_columns = ['总行权利润', '总持仓金额', '总利润百分比', '加权平均成本价', '加权平均执行价', '汇总盈利平衡价', '汇总亏损平衡价', '总合同数量']
            for col in summary_numeric_columns:
                if col in summary.columns:
                    summary[col] = summary[col].round(2)
            summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
        elif format.lower() == 'excel':
            excel_path = os.path.join(output_dir, f"{filename}.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 对详细结果进行四舍五入到2位小数
                detailed_results = self.results.copy()
                numeric_columns = ['执行价', '期权价格', '每股均价', '行权利润', '利润百分比', '盈利平衡价', '亏损平衡价', '持仓总金额']
                for col in numeric_columns:
                    if col in detailed_results.columns:
                        detailed_results[col] = detailed_results[col].round(2)
                detailed_results.to_excel(writer, sheet_name='详细分析', index=False)
                
                # 对汇总结果进行四舍五入到2位小数
                summary = self.aggregate_by_stock()
                summary_numeric_columns = ['总行权利润', '总持仓金额', '总利润百分比', '加权平均成本价', '加权平均执行价', '汇总盈利平衡价', '汇总亏损平衡价', '总合同数量']
                for col in summary_numeric_columns:
                    if col in summary.columns:
                        summary[col] = summary[col].round(2)
                summary.to_excel(writer, sheet_name='股票汇总', index=False)
        
        print(f"结果已保存到 {output_dir}/")
        return output_dir
    
    def plot_breakeven_analysis(self, save_plot: bool = True, output_dir: str = None) -> None:
        """绘制盈亏平衡分析图表"""
        if self.results is None:
            self.analyze_all_options()
        
        summary = self.aggregate_by_stock()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # 图表1: 利润百分比
        colors = ['green' if x > 0 else 'red' for x in summary['总利润百分比']]
        bars1 = ax1.bar(summary['股票代码'], summary['总利润百分比'], color=colors, alpha=0.7)
        ax1.set_title('各股票期权交易利润百分比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('利润百分比 (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars1, summary['总利润百分比']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.5),
                    f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 图表2: 盈亏平衡价格对比
        x_pos = np.arange(len(summary))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, summary['汇总盈利平衡价'], width, label='盈利平衡价', 
                color='lightgreen', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, summary['汇总亏损平衡价'], width, label='亏损平衡价', 
                color='lightcoral', alpha=0.7)
        
        # 为盈利平衡价添加价格标注
        for bar, value in zip(bars1, summary['汇总盈利平衡价']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=6)
        
        # 为亏损平衡价添加价格标注
        for bar, value in zip(bars2, summary['汇总亏损平衡价']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=6)
        
        # 添加当前成本价点
        ax2.scatter(x_pos, summary['加权平均成本价'], color='black', s=20, label='当前成本价', zorder=5)
        
        # 为当前成本价添加价格标注
        for i, (x, y) in enumerate(zip(x_pos, summary['加权平均成本价'])):
            ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
        ax2.set_title('各股票盈亏平衡价格分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('价格')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(summary['股票代码'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            if output_dir is None:
                # 创建输出目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"output/{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, "期权分析图表.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {output_dir}/期权分析图表.png")
        
        plt.show()
    
    def print_summary(self) -> None:
        """打印分析摘要"""
        if self.results is None:
            self.analyze_all_options()
        
        summary = self.aggregate_by_stock()
        
        print("\n" + "="*80)
        print("期权交易分析摘要")
        print("="*80)
        
        print(f"\n总计分析 {len(self.results)} 笔期权交易，涉及 {len(summary)} 只股票")
        
        print("\n各股票汇总分析:")
        print("-" * 80)
        for _, row in summary.iterrows():
            print(f"股票代码: {row['股票代码']}")
            print(f"  总行权利润: {row['总行权利润']:,.2f} 元")
            print(f"  总利润百分比: {row['总利润百分比']:.2f}%")
            print(f"  加权平均成本价: {row['加权平均成本价']:.2f} 元")
            print(f"  汇总盈利平衡价: {row['汇总盈利平衡价']:.2f} 元")
            print(f"  汇总亏损平衡价: {row['汇总亏损平衡价']:.2f} 元")
            print(f"  总合同数量: {row['总合同数量']:.0f} 份")
            print()
        
        # 总体统计
        total_profit = summary['总行权利润'].sum()
        total_investment = summary['总持仓金额'].sum()
        overall_profit_percentage = (total_profit / total_investment) * 100
        
        print("总体统计:")
        print("-" * 40)
        print(f"总行权利润: {total_profit:,.2f} 元")
        print(f"总持仓金额: {total_investment:,.2f} 元")
        print(f"总体利润百分比: {overall_profit_percentage:.2f}%")
        print("="*80)


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    sample_data = {
        '股票代码': ['AAPL', 'AAPL', 'TSLA', 'TSLA', 'MSFT'],
        '期权类型': ['sell call', 'sell put', 'sell call', 'sell put', 'sell call'],
        '执行日': ['2024-03-15', '2024-03-15', '2024-04-19', '2024-04-19', '2024-05-17'],
        '执行价': [180, 160, 250, 220, 400],
        '期权价格': [500, 300, 800, 600, 1200],
        '每股均价': [175, 175, 240, 240, 380],
        '合同数量': [1, 1, 2, 1, 1]
    }
    
    return pd.DataFrame(sample_data)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='期权交易分析工具')
    parser.add_argument('--input', '-i', help='输入文件路径 (CSV或Excel)')
    parser.add_argument('--output', '-o', help='输出文件名前缀')
    parser.add_argument('--format', '-f', choices=['csv', 'excel'], default='csv', help='输出格式')
    parser.add_argument('--plot', '-p', action='store_true', help='生成图表')
    parser.add_argument('--create-sample', '-s', action='store_true', help='创建示例数据文件')
    
    args = parser.parse_args()
    
    analyzer = OptionsAnalyzer()
    
    if args.create_sample:
        # 创建示例数据
        sample_data = create_sample_data()
        sample_data.to_csv('示例期权数据.csv', index=False, encoding='utf-8-sig')
        print("已创建示例数据文件: 示例期权数据.csv")
        print("请编辑此文件，填入你的实际期权交易数据")
        return
    
    if args.input:
        # 加载数据
        analyzer.load_data(args.input)
        
        # 执行分析
        analyzer.analyze_all_options()
        
        # 打印摘要
        analyzer.print_summary()
        
        # 保存结果
        if args.output:
            output_dir = analyzer.save_results(args.output, args.format)
        else:
            output_dir = analyzer.save_results(format=args.format)
        
        # 生成图表
        if args.plot:
            analyzer.plot_breakeven_analysis(output_dir=output_dir)
    
    else:
        print("请提供输入文件路径，或使用 --create-sample 创建示例数据文件")
        print("使用示例:")
        print("  python options_analyzer.py --create-sample")
        print("  python options_analyzer.py --input 示例期权数据.csv --plot")


if __name__ == "__main__":
    main() 