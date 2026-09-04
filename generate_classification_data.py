"""
生成分类任务测试数据
包含多种类型：二分类、多分类，适合机器学习和深度学习模型测试
"""
import numpy as np
import pandas as pd
import os

def generate_classification_data(output_dir="sample_data"):
    """生成分类测试数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # ========== 1. 设备故障诊断数据集（二分类：正常/故障）============
    n_samples = 1000
    features = ['温度', '压力', '振动', '转速', '功率', '电流', '电压', '噪声']
    
    # 正常状态
    normal_data = pd.DataFrame({
        '温度': np.random.normal(65, 5, 400),
        '压力': np.random.normal(100, 8, 400),
        '振动': np.random.normal(0.5, 0.1, 400),
        '转速': np.random.normal(1500, 50, 400),
        '功率': np.random.normal(75, 5, 400),
        '电流': np.random.normal(10, 1, 400),
        '电压': np.random.normal(220, 5, 400),
        '噪声': np.random.normal(55, 3, 400),
        '状态': '正常'
    })
    
    # 过热状态
    overheat_data = pd.DataFrame({
        '温度': np.random.normal(85, 8, 200),
        '压力': np.random.normal(105, 10, 200),
        '振动': np.random.normal(0.8, 0.2, 200),
        '转速': np.random.normal(1450, 60, 200),
        '功率': np.random.normal(72, 6, 200),
        '电流': np.random.normal(12, 1.5, 200),
        '电压': np.random.normal(215, 8, 200),
        '噪声': np.random.normal(65, 5, 200),
        '状态': '过热'
    })
    
    # 振动异常
    vibration_data = pd.DataFrame({
        '温度': np.random.normal(68, 6, 200),
        '压力': np.random.normal(98, 9, 200),
        '振动': np.random.normal(1.5, 0.3, 200),
        '转速': np.random.normal(1400, 80, 200),
        '功率': np.random.normal(70, 7, 200),
        '电流': np.random.normal(11, 1.2, 200),
        '电压': np.random.normal(218, 6, 200),
        '噪声': np.random.normal(75, 8, 200),
        '状态': '振动异常'
    })
    
    # 合并
    fault_data = pd.concat([normal_data, overheat_data, vibration_data], ignore_index=True)
    fault_data = fault_data.sample(frac=1).reset_index(drop=True)
    fault_data.to_csv(f"{output_dir}/设备故障诊断.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 设备故障诊断.csv ({len(fault_data)} 样本, 3类别)")
    
    # ========== 2. 产品质量分类数据集（多分类：优/良/中/差）============
    n_samples = 800
    
    # 特征：尺寸精度、表面光洁度、硬度、强度、耐腐蚀性、韧性
    features = ['尺寸精度', '表面光洁度', '硬度', '强度', '耐腐蚀性', '韧性', '密度', '弹性模量']
    
    # 优等品
    excellent = pd.DataFrame({
        '尺寸精度': np.random.normal(0.99, 0.01, 200),
        '表面光洁度': np.random.normal(0.95, 0.03, 200),
        '硬度': np.random.normal(65, 3, 200),
        '强度': np.random.normal(500, 30, 200),
        '耐腐蚀性': np.random.normal(0.92, 0.05, 200),
        '韧性': np.random.normal(85, 5, 200),
        '密度': np.random.normal(7.85, 0.05, 200),
        '弹性模量': np.random.normal(210, 10, 200),
        '质量等级': '优等品'
    })
    
    # 良好
    good = pd.DataFrame({
        '尺寸精度': np.random.normal(0.95, 0.02, 200),
        '表面光洁度': np.random.normal(0.88, 0.05, 200),
        '硬度': np.random.normal(60, 5, 200),
        '强度': np.random.normal(450, 40, 200),
        '耐腐蚀性': np.random.normal(0.85, 0.08, 200),
        '韧性': np.random.normal(78, 8, 200),
        '密度': np.random.normal(7.80, 0.08, 200),
        '弹性模量': np.random.normal(205, 15, 200),
        '质量等级': '良好'
    })
    
    # 中等
    medium = pd.DataFrame({
        '尺寸精度': np.random.normal(0.88, 0.03, 200),
        '表面光洁度': np.random.normal(0.78, 0.08, 200),
        '硬度': np.random.normal(52, 7, 200),
        '强度': np.random.normal(380, 50, 200),
        '耐腐蚀性': np.random.normal(0.75, 0.1, 200),
        '韧性': np.random.normal(68, 10, 200),
        '密度': np.random.normal(7.70, 0.12, 200),
        '弹性模量': np.random.normal(195, 20, 200),
        '质量等级': '中等'
    })
    
    # 差
    poor = pd.DataFrame({
        '尺寸精度': np.random.normal(0.75, 0.05, 200),
        '表面光洁度': np.random.normal(0.60, 0.1, 200),
        '硬度': np.random.normal(40, 10, 200),
        '强度': np.random.normal(280, 60, 200),
        '耐腐蚀性': np.random.normal(0.55, 0.15, 200),
        '韧性': np.random.normal(50, 15, 200),
        '密度': np.random.normal(7.50, 0.2, 200),
        '弹性模量': np.random.normal(170, 30, 200),
        '质量等级': '差'
    })
    
    quality_data = pd.concat([excellent, good, medium, poor], ignore_index=True)
    quality_data = quality_data.sample(frac=1).reset_index(drop=True)
    quality_data.to_csv(f"{output_dir}/产品质量分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 产品质量分类.csv ({len(quality_data)} 样本, 4类别)")
    
    # ========== 3. 鸢尾花数据集（经典三分类）============
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['品种'] = iris.target_names[iris.target]
    iris_df.to_csv(f"{output_dir}/鸢尾花分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 鸢尾花分类.csv ({len(iris_df)} 样本, 3类别)")
    
    # ========== 4. 心脏病诊断数据集（二分类）============
    n_samples = 600
    
    features = ['年龄', '血压', '胆固醇', '血糖', '心率', '运动耐量', '最大心率', 'ST段压低']
    
    # 无心脏病
    no_disease = pd.DataFrame({
        '年龄': np.random.normal(45, 10, 300),
        '血压': np.random.normal(120, 15, 300),
        '胆固醇': np.random.normal(200, 30, 300),
        '血糖': np.random.normal(90, 10, 300),
        '心率': np.random.normal(72, 8, 300),
        '运动耐量': np.random.normal(8, 1.5, 300),
        '最大心率': np.random.normal(170, 15, 300),
        'ST段压低': np.random.normal(0.5, 0.2, 300),
        '诊断结果': '无心脏病'
    })
    
    # 有心脏病
    has_disease = pd.DataFrame({
        '年龄': np.random.normal(55, 12, 300),
        '血压': np.random.normal(145, 20, 300),
        '胆固醇': np.random.normal(260, 40, 300),
        '血糖': np.random.normal(120, 20, 300),
        '心率': np.random.normal(85, 12, 300),
        '运动耐量': np.random.normal(5, 2, 300),
        '最大心率': np.random.normal(140, 25, 300),
        'ST段压低': np.random.normal(2.5, 1.0, 300),
        '诊断结果': '有心脏病'
    })
    
    heart_data = pd.concat([no_disease, has_disease], ignore_index=True)
    heart_data = heart_data.sample(frac=1).reset_index(drop=True)
    heart_data.to_csv(f"{output_dir}/心脏病诊断.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 心脏病诊断.csv ({len(heart_data)} 样本, 2类别)")
    
    # ========== 5. 客户流失分类（二分类）============
    n_samples = 500
    
    features = ['月消费', '在网时长', '投诉次数', '流量使用', '语音通话', '增值服务数', '满意度评分', '欠费金额']
    
    # 留存客户
    retained = pd.DataFrame({
        '月消费': np.random.normal(120, 30, 250),
        '在网时长': np.random.normal(36, 12, 250),
        '投诉次数': np.random.normal(0.5, 0.8, 250),
        '流量使用': np.random.normal(15, 5, 250),
        '语音通话': np.random.normal(300, 100, 250),
        '增值服务数': np.random.normal(3, 1, 250),
        '满意度评分': np.random.normal(4.2, 0.6, 250),
        '欠费金额': np.random.normal(20, 15, 250),
        '客户状态': '留存'
    })
    
    # 流失客户
    churned = pd.DataFrame({
        '月消费': np.random.normal(80, 25, 250),
        '在网时长': np.random.normal(12, 8, 250),
        '投诉次数': np.random.normal(3.5, 2, 250),
        '流量使用': np.random.normal(8, 4, 250),
        '语音通话': np.random.normal(150, 80, 250),
        '增值服务数': np.random.normal(1.2, 0.8, 250),
        '满意度评分': np.random.normal(2.5, 0.8, 250),
        '欠费金额': np.random.normal(80, 40, 250),
        '客户状态': '流失'
    })
    
    churn_data = pd.concat([retained, churned], ignore_index=True)
    churn_data = churn_data.sample(frac=1).reset_index(drop=True)
    churn_data.to_csv(f"{output_dir}/客户流失分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 客户流失分类.csv ({len(churn_data)} 样本, 2类别)")
    
    # ========== 6. 机械振动模式分类（时序特征，适合LSTM/GRU）============
    n_samples = 400
    
    # 为每个样本生成时序特征（模拟100个时间步）
    time_steps = 50
    
    def generate_time_series(label, n_samples, base_params):
        """生成时序数据"""
        data_list = []
        for i in range(n_samples):
            t = np.linspace(0, 1, time_steps)
            # 基础信号
            signal = base_params[0] * np.sin(2 * np.pi * base_params[1] * t + base_params[2])
            # 添加噪声
            noise = np.random.normal(0, base_params[3], time_steps)
            signal = signal + noise
            # 添加一些随机波动
            signal = signal + np.random.normal(0, 0.1, time_steps)
            
            # 提取统计特征
            row = {
                '均值': np.mean(signal),
                '标准差': np.std(signal),
                '最大值': np.max(signal),
                '最小值': np.min(signal),
                '峰峰值': np.max(signal) - np.min(signal),
                '均方根': np.sqrt(np.mean(signal**2)),
                '偏度': np.mean(((signal - np.mean(signal)) / (np.std(signal) + 1e-10))**3),
                '峰度': np.mean(((signal - np.mean(signal)) / (np.std(signal) + 1e-10))**4),
                '波形指数': np.sqrt(np.mean(signal**2)) / (np.mean(np.abs(signal)) + 1e-10),
                '峰值指数': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-10),
                '状态': label
            }
            data_list.append(row)
        return pd.DataFrame(data_list)
    
    # 正常振动
    normal_vib = generate_time_series('正常', 100, [1.0, 5.0, 0, 0.1])
    # 磨损故障
    wear = generate_time_series('磨损', 100, [1.5, 8.0, 0.5, 0.2])
    # 不平衡
    imbalance = generate_time_series('不平衡', 100, [2.0, 3.0, 1.0, 0.15])
    # 轴承故障
    bearing = generate_time_series('轴承故障', 100, [2.5, 12.0, 0, 0.3])
    
    vibration_data = pd.concat([normal_vib, wear, imbalance, bearing], ignore_index=True)
    vibration_data = vibration_data.sample(frac=1).reset_index(drop=True)
    vibration_data.to_csv(f"{output_dir}/机械振动模式分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 机械振动模式分类.csv ({len(vibration_data)} 样本, 4类别)")
    
    # ========== 7. 糖尿病风险预测（二分类）============
    n_samples = 500
    
    features = ['年龄', '血糖', '血压', 'BMI', '胆固醇', '运动时长', '饮食习惯', '家族病史']
    
    # 低风险
    low_risk = pd.DataFrame({
        '年龄': np.random.normal(35, 8, 200),
        '血糖': np.random.normal(90, 10, 200),
        '血压': np.random.normal(110, 12, 200),
        'BMI': np.random.normal(22, 3, 200),
        '胆固醇': np.random.normal(180, 25, 200),
        '运动时长': np.random.normal(60, 15, 200),
        '饮食习惯': np.random.normal(7, 2, 200),
        '家族病史': np.random.choice([0, 1], 200, p=[0.9, 0.1]),
        '风险等级': '低风险'
    })
    
    # 中风险
    medium_risk = pd.DataFrame({
        '年龄': np.random.normal(50, 10, 150),
        '血糖': np.random.normal(115, 15, 150),
        '血压': np.random.normal(130, 18, 150),
        'BMI': np.random.normal(26, 4, 150),
        '胆固醇': np.random.normal(220, 30, 150),
        '运动时长': np.random.normal(35, 12, 150),
        '饮食习惯': np.random.normal(5, 2, 150),
        '家族病史': np.random.choice([0, 1], 150, p=[0.7, 0.3]),
        '风险等级': '中风险'
    })
    
    # 高风险
    high_risk = pd.DataFrame({
        '年龄': np.random.normal(60, 12, 150),
        '血糖': np.random.normal(150, 25, 150),
        '血压': np.random.normal(155, 22, 150),
        'BMI': np.random.normal(30, 5, 150),
        '胆固醇': np.random.normal(270, 40, 150),
        '运动时长': np.random.normal(15, 10, 150),
        '饮食习惯': np.random.normal(3, 1.5, 150),
        '家族病史': np.random.choice([0, 1], 150, p=[0.4, 0.6]),
        '风险等级': '高风险'
    })
    
    diabetes_data = pd.concat([low_risk, medium_risk, high_risk], ignore_index=True)
    diabetes_data = diabetes_data.sample(frac=1).reset_index(drop=True)
    diabetes_data.to_csv(f"{output_dir}/糖尿病风险预测.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 糖尿病风险预测.csv ({len(diabetes_data)} 样本, 3类别)")
    
    # ========== 8. 信用评分分类（五分类：A/B/C/D/E）============
    n_samples = 600
    
    features = ['年收入', '负债率', '信用记录年限', '贷款金额', '还款记录', '资产价值', '职业稳定性', '教育程度']
    
    # A级（最优）
    grade_a = pd.DataFrame({
        '年收入': np.random.normal(80, 20, 120),
        '负债率': np.random.normal(0.15, 0.05, 120),
        '信用记录年限': np.random.normal(10, 3, 120),
        '贷款金额': np.random.normal(20, 8, 120),
        '还款记录': np.random.normal(95, 3, 120),
        '资产价值': np.random.normal(500, 100, 120),
        '职业稳定性': np.random.normal(8, 2, 120),
        '教育程度': np.random.normal(16, 2, 120),
        '信用等级': 'A'
    })
    
    # B级
    grade_b = pd.DataFrame({
        '年收入': np.random.normal(60, 15, 120),
        '负债率': np.random.normal(0.25, 0.08, 120),
        '信用记录年限': np.random.normal(7, 2, 120),
        '贷款金额': np.random.normal(30, 10, 120),
        '还款记录': np.random.normal(88, 5, 120),
        '资产价值': np.random.normal(350, 80, 120),
        '职业稳定性': np.random.normal(6, 2, 120),
        '教育程度': np.random.normal(14, 2, 120),
        '信用等级': 'B'
    })
    
    # C级
    grade_c = pd.DataFrame({
        '年收入': np.random.normal(45, 12, 120),
        '负债率': np.random.normal(0.40, 0.10, 120),
        '信用记录年限': np.random.normal(5, 2, 120),
        '贷款金额': np.random.normal(40, 15, 120),
        '还款记录': np.random.normal(80, 8, 120),
        '资产价值': np.random.normal(250, 60, 120),
        '职业稳定性': np.random.normal(5, 2, 120),
        '教育程度': np.random.normal(12, 2, 120),
        '信用等级': 'C'
    })
    
    # D级
    grade_d = pd.DataFrame({
        '年收入': np.random.normal(30, 10, 120),
        '负债率': np.random.normal(0.55, 0.12, 120),
        '信用记录年限': np.random.normal(3, 2, 120),
        '贷款金额': np.random.normal(50, 20, 120),
        '还款记录': np.random.normal(70, 10, 120),
        '资产价值': np.random.normal(150, 50, 120),
        '职业稳定性': np.random.normal(3, 2, 120),
        '教育程度': np.random.normal(10, 2, 120),
        '信用等级': 'D'
    })
    
    # E级（最差）
    grade_e = pd.DataFrame({
        '年收入': np.random.normal(20, 8, 120),
        '负债率': np.random.normal(0.70, 0.15, 120),
        '信用记录年限': np.random.normal(1.5, 1, 120),
        '贷款金额': np.random.normal(60, 25, 120),
        '还款记录': np.random.normal(55, 15, 120),
        '资产价值': np.random.normal(80, 40, 120),
        '职业稳定性': np.random.normal(2, 1.5, 120),
        '教育程度': np.random.normal(9, 2, 120),
        '信用等级': 'E'
    })
    
    credit_data = pd.concat([grade_a, grade_b, grade_c, grade_d, grade_e], ignore_index=True)
    credit_data = credit_data.sample(frac=1).reset_index(drop=True)
    credit_data.to_csv(f"{output_dir}/信用评分分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 信用评分分类.csv ({len(credit_data)} 样本, 5类别)")
    
    # ========== 9. 图像质量评估（四分类）============
    n_samples = 400
    
    features = ['清晰度', '对比度', '亮度', '色彩饱和度', '噪点', '伪影', '边缘锐度', '动态范围']
    
    # 优秀
    excellent_img = pd.DataFrame({
        '清晰度': np.random.normal(90, 5, 100),
        '对比度': np.random.normal(85, 8, 100),
        '亮度': np.random.normal(128, 15, 100),
        '色彩饱和度': np.random.normal(80, 10, 100),
        '噪点': np.random.normal(2, 1, 100),
        '伪影': np.random.normal(1, 0.5, 100),
        '边缘锐度': np.random.normal(90, 5, 100),
        '动态范围': np.random.normal(8, 1, 100),
        '质量等级': '优秀'
    })
    
    # 良好
    good_img = pd.DataFrame({
        '清晰度': np.random.normal(75, 8, 100),
        '对比度': np.random.normal(70, 12, 100),
        '亮度': np.random.normal(120, 20, 100),
        '色彩饱和度': np.random.normal(65, 15, 100),
        '噪点': np.random.normal(5, 2, 100),
        '伪影': np.random.normal(3, 1.5, 100),
        '边缘锐度': np.random.normal(75, 8, 100),
        '动态范围': np.random.normal(6.5, 1.5, 100),
        '质量等级': '良好'
    })
    
    # 一般
    medium_img = pd.DataFrame({
        '清晰度': np.random.normal(55, 10, 100),
        '对比度': np.random.normal(50, 15, 100),
        '亮度': np.random.normal(100, 30, 100),
        '色彩饱和度': np.random.normal(45, 18, 100),
        '噪点': np.random.normal(12, 4, 100),
        '伪影': np.random.normal(8, 3, 100),
        '边缘锐度': np.random.normal(55, 12, 100),
        '动态范围': np.random.normal(5, 2, 100),
        '质量等级': '一般'
    })
    
    # 差
    poor_img = pd.DataFrame({
        '清晰度': np.random.normal(30, 12, 100),
        '对比度': np.random.normal(30, 15, 100),
        '亮度': np.random.normal(80, 40, 100),
        '色彩饱和度': np.random.normal(25, 15, 100),
        '噪点': np.random.normal(25, 8, 100),
        '伪影': np.random.normal(18, 6, 100),
        '边缘锐度': np.random.normal(30, 15, 100),
        '动态范围': np.random.normal(3, 1.5, 100),
        '质量等级': '差'
    })
    
    img_quality_data = pd.concat([excellent_img, good_img, medium_img, poor_img], ignore_index=True)
    img_quality_data = img_quality_data.sample(frac=1).reset_index(drop=True)
    img_quality_data.to_csv(f"{output_dir}/图像质量评估.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 图像质量评估.csv ({len(img_quality_data)} 样本, 4类别)")
    
    # ========== 10. 邮件垃圾邮件分类（二分类）============
    n_samples = 500
    
    features = ['邮件长度', '链接数量', '关键词密度', '大写字母比例', '感叹号数量', '特殊字符', '发件人可信度', '主题长度']
    
    # 正常邮件
    normal_mail = pd.DataFrame({
        '邮件长度': np.random.normal(800, 300, 250),
        '链接数量': np.random.normal(2, 1.5, 250),
        '关键词密度': np.random.normal(0.03, 0.01, 250),
        '大写字母比例': np.random.normal(0.05, 0.02, 250),
        '感叹号数量': np.random.normal(0.5, 0.8, 250),
        '特殊字符': np.random.normal(1, 1, 250),
        '发件人可信度': np.random.normal(85, 10, 250),
        '主题长度': np.random.normal(40, 15, 250),
        '类型': '正常邮件'
    })
    
    # 垃圾邮件
    spam_mail = pd.DataFrame({
        '邮件长度': np.random.normal(1200, 400, 250),
        '链接数量': np.random.normal(8, 3, 250),
        '关键词密度': np.random.normal(0.15, 0.05, 250),
        '大写字母比例': np.random.normal(0.20, 0.08, 250),
        '感叹号数量': np.random.normal(5, 2.5, 250),
        '特殊字符': np.random.normal(8, 3, 250),
        '发件人可信度': np.random.normal(30, 15, 250),
        '主题长度': np.random.normal(60, 25, 250),
        '类型': '垃圾邮件'
    })
    
    spam_data = pd.concat([normal_mail, spam_mail], ignore_index=True)
    spam_data = spam_data.sample(frac=1).reset_index(drop=True)
    spam_data.to_csv(f"{output_dir}/邮件垃圾邮件分类.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 已生成: 邮件垃圾邮件分类.csv ({len(spam_data)} 样本, 2类别)")
    
    print(f"\n✅ 分类测试数据生成完成！共生成 10 个数据集，保存在 {output_dir}/ 目录下")
    
    return output_dir

if __name__ == "__main__":
    generate_classification_data()
