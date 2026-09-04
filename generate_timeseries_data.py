"""
生成时序预测测试数据
包含多种类型的时序数据：温度、股票、销售、风力发电等
"""
import numpy as np
import pandas as pd
import os

def generate_time_series_data(output_dir="sample_data"):
    """生成时序预测测试数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # ========== 1. 温度时序预测 ==========
    n_samples = 500
    
    # 模拟一年温度数据（带趋势和季节性）
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    t = np.arange(n_samples)
    
    # 温度 = 趋势 + 季节性 + 噪声
    trend = 15 + 0.01 * t  # 缓慢上升趋势
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # 季节性
    noise = np.random.normal(0, 2, n_samples)
    temperature = trend + seasonal + noise
    
    temp_data = pd.DataFrame({
        '日期': dates,
        '温度': temperature,
        '湿度': 60 + 15 * np.sin(2 * np.pi * t / 365 + 1) + np.random.normal(0, 5, n_samples),
        '气压': 1013 + 5 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 3, n_samples),
        '风速': np.abs(5 + 3 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2, n_samples))
    })
    temp_data.to_csv(f"{output_dir}/温度时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 温度时序预测.csv ({len(temp_data)} 样本)")
    
    # ========== 2. 电力负荷时序预测 ==========
    n_samples = 720  # 30天 * 24小时
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    t = np.arange(n_samples)
    hour = t % 24
    day = t // 24
    
    # 电力负荷 = 每日周期 + 每周周期 + 趋势 + 噪声
    daily_pattern = 100 + 50 * np.sin(2 * np.pi * (hour - 6) / 24)  # 每日周期
    weekly_pattern = 20 * np.sin(2 * np.pi * day / 7)  # 每周周期
    trend = 0.05 * t  # 增长趋势
    noise = np.random.normal(0, 10, n_samples)
    load = daily_pattern + weekly_pattern + trend + noise
    
    power_data = pd.DataFrame({
        '时间': dates,
        '电力负荷': load,
        '温度': 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3, n_samples),
        '湿度': 70 + 10 * np.sin(2 * np.pi * hour / 24 + 2) + np.random.normal(0, 5, n_samples),
        '是否工作日': [(d.weekday() < 5) * 1 for d in dates]
    })
    power_data.to_csv(f"{output_dir}/电力负荷时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 电力负荷时序预测.csv ({len(power_data)} 样本)")
    
    # ========== 3. 股票价格时序预测 ==========
    n_samples = 250  # 一年交易日
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='B')  # 工作日
    t = np.arange(n_samples)
    
    # 使用几何布朗运动模拟股价
    returns = np.random.normal(0.001, 0.02, n_samples)
    returns[0] = 0
    log_price = np.cumsum(returns)
    price = 100 * np.exp(log_price)
    
    stock_data = pd.DataFrame({
        '日期': dates,
        '开盘价': price * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
        '最高价': price * (1 + np.random.uniform(0, 0.03, n_samples)),
        '最低价': price * (1 - np.random.uniform(0, 0.03, n_samples)),
        '收盘价': price,
        '成交量': np.random.uniform(1000000, 5000000, n_samples)
    })
    stock_data.to_csv(f"{output_dir}/股票价格时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 股票价格时序预测.csv ({len(stock_data)} 样本)")
    
    # ========== 4. 销售额时序预测 ==========
    n_samples = 365  # 一年
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    t = np.arange(n_samples)
    
    # 销售 = 趋势 + 周末效应 + 季节性 + 噪声
    trend = 5000 + 10 * t  # 增长趋势
    weekend_effect = np.array([0 if d.weekday() < 5 else -2000 for d in dates])  # 周末降低
    seasonal = 1000 * np.sin(2 * np.pi * t / 30)  # 月度周期
    noise = np.random.normal(0, 500, n_samples)
    sales = trend + weekend_effect + seasonal + noise
    
    sales_data = pd.DataFrame({
        '日期': dates,
        '销售额': sales,
        '顾客数': (sales / 50 + np.random.normal(0, 20, n_samples)).astype(int),
        '促销': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        '气温': 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 5, n_samples)
    })
    sales_data.to_csv(f"{output_dir}/销售额时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 销售额时序预测.csv ({len(sales_data)} 样本)")
    
    # ========== 5. 风力发电时序预测 ==========
    n_samples = 720  # 30天 * 24小时
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    t = np.arange(n_samples)
    hour = t % 24
    
    # 风速 = 每日周期 + 随机波动
    base_wind = 8 + 3 * np.sin(2 * np.pi * hour / 24 + 3)
    wind_gust = np.random.normal(0, 2, n_samples)
    wind_speed = np.maximum(base_wind + wind_gust, 0)
    
    # 发电量与风速的三次方成正比（风能公式）
    power = 0.5 * 1.225 * np.pi * 50**2 * (wind_speed**3) / 1e6  # MW
    power = np.clip(power, 0, 3) + np.random.normal(0, 0.1, n_samples)  # 添加噪声
    
    wind_data = pd.DataFrame({
        '时间': dates,
        '风速': wind_speed,
        '发电量': power,
        '风向': (180 + 45 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 20, n_samples)) % 360,
        '气温': 15 + 8 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3, n_samples),
        '空气密度': 1.225 - 0.004 * (15 + 8 * np.sin(2 * np.pi * t / 24))  # 温度影响密度
    })
    wind_data.to_csv(f"{output_dir}/风力发电时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 风力发电时序预测.csv ({len(wind_data)} 样本)")
    
    # ========== 6. 交通流量时序预测 ==========
    n_samples = 288  # 12天 * 24小时
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    t = np.arange(n_samples)
    hour = t % 24
    day_type = [(d.weekday() < 5) for d in dates]  # 工作日/周末
    
    # 交通流量
    traffic = np.zeros(n_samples)
    for i, (h, is_workday) in enumerate(zip(hour, day_type)):
        if is_workday:
            # 工作日：早高峰7-9，晚高峰17-19
            if 7 <= h <= 9:
                traffic[i] = 2000 + 500 * np.sin(np.pi * (h - 7) / 2)
            elif 17 <= h <= 19:
                traffic[i] = 1800 + 400 * np.sin(np.pi * (h - 17) / 2)
            else:
                traffic[i] = 800 + 200 * np.sin(np.pi * h / 24)
        else:
            # 周末：中午较高
            if 10 <= h <= 16:
                traffic[i] = 1200 + 300 * np.sin(np.pi * (h - 10) / 6)
            else:
                traffic[i] = 500 + 100 * np.sin(np.pi * h / 24)
    
    traffic = traffic + np.random.normal(0, 100, n_samples)
    
    traffic_data = pd.DataFrame({
        '时间': dates,
        '车流量': traffic.astype(int),
        '平均车速': 60 - 0.005 * traffic + np.random.normal(0, 5, n_samples),
        '拥堵指数': traffic / 2500 + np.random.normal(0, 0.1, n_samples),
        '天气': np.random.choice(['晴', '阴', '雨'], n_samples, p=[0.6, 0.3, 0.1])
    })
    traffic_data.to_csv(f"{output_dir}/交通流量时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 交通流量时序预测.csv ({len(traffic_data)} 样本)")
    
    # ========== 7. 传感器时序预测（用于设备预测性维护）==========
    n_samples = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='min')
    t = np.arange(n_samples)
    
    # 模拟设备退化过程
    # 温度、振动、功率逐渐上升（设备老化）
    degradation = t / n_samples * 50  # 退化因子
    
    temperature = 50 + 10 * np.sin(2 * np.pi * t / 60) + degradation * 0.3 + np.random.normal(0, 2, n_samples)
    vibration = 0.5 + 0.2 * degradation + 0.1 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.05, n_samples)
    power = 100 + 5 * np.sin(2 * np.pi * t / 120) + degradation * 0.1 + np.random.normal(0, 2, n_samples)
    rpm = 1500 + 20 * np.sin(2 * np.pi * t / 60) - degradation * 5 + np.random.normal(0, 10, n_samples)
    
    sensor_data = pd.DataFrame({
        '时间': dates,
        '温度': temperature,
        '振动': vibration,
        '功率': power,
        '转速': rpm,
        '故障概率': degradation / 50  # 退化因子作为故障概率
    })
    sensor_data.to_csv(f"{output_dir}/设备传感器时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 设备传感器时序预测.csv ({len(sensor_data)} 样本)")
    
    # ========== 8. 网络流量时序预测 ==========
    n_samples = 504  # 21天 * 24小时
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    t = np.arange(n_samples)
    hour = t % 24
    
    # 网络流量 = 每日周期 + 工作日/周末差异 + 噪声
    base_traffic = 500 + 200 * np.sin(2 * np.pi * (hour - 2) / 24)  # 凌晨低，白天高
    day_factor = np.array([1.5 if d.weekday() < 5 else 0.8 for d in dates])
    traffic = base_traffic * day_factor + np.random.normal(0, 50, n_samples)
    traffic = np.maximum(traffic, 50)
    
    network_data = pd.DataFrame({
        '时间': dates,
        '入流量': traffic,
        '出流量': traffic * 0.8 + np.random.normal(0, 30, n_samples),
        '延迟': 10 + 5 * np.sin(2 * np.pi * hour / 24) + traffic / 100 + np.random.normal(0, 2, n_samples),
        '丢包率': 0.01 + 0.005 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 0.005, n_samples),
        'CPU使用率': 30 + 20 * np.sin(2 * np.pi * hour / 24) + traffic / 30 + np.random.normal(0, 5, n_samples)
    })
    network_data.to_csv(f"{output_dir}/网络流量时序预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 网络流量时序预测.csv ({len(network_data)} 样本)")
    
    print(f"\n✅ 时序预测测试数据生成完成！共生成 8 个数据集，保存在 {output_dir}/ 目录下")
    
    return output_dir

if __name__ == "__main__":
    generate_time_series_data()
