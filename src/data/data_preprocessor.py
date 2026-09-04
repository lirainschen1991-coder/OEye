import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.selected_features = None
    
    def clean_data(self, df, drop_na_threshold=0.8):
        """
        数据清洗
        """
        # 计算每列的缺失值比例
        na_ratio = df.isnull().sum() / len(df)
        # 删除缺失值比例超过阈值的列
        df_clean = df.drop(columns=na_ratio[na_ratio > drop_na_threshold].index)
        
        # 获取数值列
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # 处理无穷值和过大值
        for col in numeric_cols:
            # 将无穷值替换为NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            # 将超过1e10的值视为异常值，替换为NaN
            df_clean[col] = df_clean[col].where(df_clean[col].abs() < 1e10, np.nan)
        
        # 填充剩余的缺失值
        self.imputer = SimpleImputer(strategy='mean')
        df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
        
        return df_clean
    
    def scale_data(self, X, method='standard'):
        """
        数据缩放
        """
        X_clean = X.copy()
        
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            try:
                X_clean.loc[X_clean[col].abs() >= 1e10, col] = np.nan
            except:
                pass
        
        non_numeric_cols = [col for col in X_clean.columns if col not in numeric_cols]
        X_clean = X_clean[numeric_cols]
        
        if len(numeric_cols) == 0:
            return X_clean
        
        if X_clean.isnull().any().any():
            imputer = SimpleImputer(strategy='mean')
            X_clean = pd.DataFrame(
                imputer.fit_transform(X_clean),
                columns=X_clean.columns,
                index=X_clean.index
            )
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")
        
        X_scaled = self.scaler.fit_transform(X_clean)
        
        return pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
    
    def select_features(self, X, y, k=10, method='f_regression'):
        """
        特征选择
        """
        if method == 'f_regression':
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)

    def prepare_train_val_test(self, X, y, test_size=0.2, val_size=0.1, random_state=42,
                               time_series=False, scale_method='standard',
                               use_feature_selection=False, k=10,
                               selection_method='f_regression'):
        """
        无数据泄漏的数据准备流程：先划分，再仅在训练集fit缩放器和特征选择器。
        """
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = self.split_data(
            X, y, test_size=test_size, val_size=val_size,
            random_state=random_state, time_series=time_series
        )

        X_train_raw = self._ensure_frame(X_train_raw, X)
        X_val_raw = self._ensure_frame(X_val_raw, X) if X_val_raw is not None else None
        X_test_raw = self._ensure_frame(X_test_raw, X)

        X_train_scaled = self._fit_scale_train(X_train_raw, scale_method)
        X_val_scaled = self._transform_scale_split(X_val_raw) if X_val_raw is not None else None
        X_test_scaled = self._transform_scale_split(X_test_raw)

        selected_features = list(X_train_scaled.columns)
        if use_feature_selection:
            k = min(max(1, int(k)), X_train_scaled.shape[1])
            self._fit_feature_selector(X_train_scaled, y_train, k, selection_method)
            X_train_final = self._transform_selected_features(X_train_scaled)
            X_val_final = self._transform_selected_features(X_val_scaled) if X_val_scaled is not None else None
            X_test_final = self._transform_selected_features(X_test_scaled)
            selected_features = list(self.selected_features)
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
            self.selected_features = pd.Index(selected_features)

        return {
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'selected_features': selected_features,
            'scale_method': scale_method,
            'time_series': time_series
        }

    def _ensure_frame(self, X_part, X_reference):
        if isinstance(X_part, pd.DataFrame):
            return X_part.copy()
        columns = list(X_reference.columns) if hasattr(X_reference, 'columns') else [f'feature_{i}' for i in range(np.asarray(X_part).shape[1])]
        return pd.DataFrame(X_part, columns=columns)

    def _fit_scale_train(self, X_train, method):
        X_clean = X_train.copy().replace([np.inf, -np.inf], np.nan)
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.pipeline_numeric_cols = numeric_cols

        if not numeric_cols:
            return pd.DataFrame(index=X_train.index)

        X_numeric = X_clean[numeric_cols]
        self.pipeline_imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            self.pipeline_imputer.fit_transform(X_numeric),
            columns=numeric_cols,
            index=X_train.index
        )

        if method == 'none':
            self.scaler = None
            return X_imputed
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")

        return pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=numeric_cols,
            index=X_train.index
        )

    def _transform_scale_split(self, X_part):
        if X_part is None:
            return None
        numeric_cols = getattr(self, 'pipeline_numeric_cols', [])
        if not numeric_cols:
            return pd.DataFrame(index=X_part.index)

        X_clean = X_part.copy().replace([np.inf, -np.inf], np.nan)
        X_numeric = X_clean.reindex(columns=numeric_cols)
        X_imputed = pd.DataFrame(
            self.pipeline_imputer.transform(X_numeric),
            columns=numeric_cols,
            index=X_part.index
        )
        if self.scaler is None:
            return X_imputed
        return pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=numeric_cols,
            index=X_part.index
        )

    def _fit_feature_selector(self, X_train, y_train, k, method):
        if method == 'f_regression':
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'rfe':
            self.feature_selector = RFE(LinearRegression(), n_features_to_select=k)
        elif method == 'tree_based':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.feature_selector = SelectFromModel(estimator, max_features=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        self.feature_selector.fit(X_train, y_train)
        self.selected_features = X_train.columns[self.feature_selector.get_support()]

    def _transform_selected_features(self, X_part):
        X_selected = self.feature_selector.transform(X_part)
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X_part.index)
    
    # ==================== 异常检测和修复功能 ====================
    
    def detect_outliers(self, df, columns=None, method='iqr', threshold=3):
        """
        检测异常值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list, optional
            要检测的列，默认为所有数值列
        method : str
            检测方法: 'iqr', 'zscore', 'isolation_forest'
        threshold : float
            异常值阈值
            
        Returns:
        --------
        dict : 各列的异常值索引
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_dict = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[col].dropna().index[z_scores > threshold]
                
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                preds = iso_forest.fit_predict(df[[col]].dropna())
                outliers = df[[col]].dropna().index[preds == -1]
            
            outliers_dict[col] = outliers.tolist()
        
        return outliers_dict
    
    def repair_outliers(self, df, columns=None, method='interpolation', outliers_dict=None):
        """
        修复异常值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list, optional
            要修复的列
        method : str
            修复方法: 'interpolation', 'mean', 'median', 'forward_fill', 'backward_fill'
        outliers_dict : dict, optional
            预检测的异常值字典
            
        Returns:
        --------
        DataFrame : 修复后的数据
        """
        df_repaired = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if outliers_dict is None:
            outliers_dict = self.detect_outliers(df_repaired, columns)
        
        for col in columns:
            outlier_indices = outliers_dict.get(col, [])
            
            if len(outlier_indices) == 0:
                continue
            
            if method == 'interpolation':
                df_repaired.loc[outlier_indices, col] = np.nan
                df_repaired[col] = df_repaired[col].interpolate(method='linear')
            elif method == 'mean':
                mean_val = df_repaired[col].mean()
                df_repaired.loc[outlier_indices, col] = mean_val
            elif method == 'median':
                median_val = df_repaired[col].median()
                df_repaired.loc[outlier_indices, col] = median_val
            elif method == 'forward_fill':
                df_repaired.loc[outlier_indices, col] = np.nan
                df_repaired[col] = df_repaired[col].fillna(method='ffill')
            elif method == 'backward_fill':
                df_repaired.loc[outlier_indices, col] = np.nan
                df_repaired[col] = df_repaired[col].fillna(method='bfill')
        
        return df_repaired
    
    def smooth_data(self, df, columns=None, method='savgol', window_length=5, polyorder=2):
        """
        数据平滑处理
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list, optional
            要平滑的列
        method : str
            平滑方法: 'savgol', 'moving_average', 'exponential'
        window_length : int
            窗口长度
        polyorder : int
            多项式阶数（用于Savgol滤波器）
            
        Returns:
        --------
        DataFrame : 平滑后的数据
        """
        df_smooth = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == 'savgol':
                if window_length % 2 == 0:
                    window_length += 1
                df_smooth[col] = savgol_filter(df_smooth[col], window_length, polyorder)
            elif method == 'moving_average':
                df_smooth[col] = df_smooth[col].rolling(window=window_length, center=True).mean()
                df_smooth[col] = df_smooth[col].fillna(method='bfill').fillna(method='ffill')
            elif method == 'exponential':
                df_smooth[col] = df_smooth[col].ewm(span=window_length).mean()
        
        return df_smooth
    
    # ==================== 数据增强功能 ====================
    
    def augment_data(self, df, target_col=None, method='noise', noise_factor=0.01, n_augmentations=1):
        """
        数据增强
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        target_col : str, optional
            目标列名称（不对目标列添加噪声）
        method : str
            增强方法: 'noise', 'scaling', 'time_warp', 'permutation'
        noise_factor : float
            噪声因子
        n_augmentations : int
            增强次数
            
        Returns:
        --------
        DataFrame : 增强后的数据（包含原始数据）
        """
        df_aug_list = [df.copy()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for _ in range(n_augmentations):
            df_aug = df.copy()
            
            if method == 'noise':
                for col in numeric_cols:
                    noise = np.random.normal(0, df[col].std() * noise_factor, len(df))
                    df_aug[col] = df_aug[col] + noise
                    
            elif method == 'scaling':
                for col in numeric_cols:
                    scale = np.random.uniform(0.9, 1.1)
                    df_aug[col] = df_aug[col] * scale
                    
            elif method == 'time_warp':
                # 时间扭曲 - 对时间序列进行随机拉伸/压缩
                warp_factor = np.random.uniform(0.8, 1.2)
                for col in numeric_cols:
                    indices = np.linspace(0, len(df) - 1, len(df))
                    warped_indices = np.clip(indices * warp_factor, 0, len(df) - 1)
                    df_aug[col] = np.interp(indices, warped_indices, df_aug[col])
                    
            elif method == 'permutation':
                # 随机置换小段序列
                segment_size = max(1, len(df) // 10)
                for col in numeric_cols:
                    segments = [df_aug[col].iloc[i:i+segment_size] for i in range(0, len(df), segment_size)]
                    np.random.shuffle(segments)
                    df_aug[col] = pd.concat(segments).reset_index(drop=True)
            
            df_aug_list.append(df_aug)
        
        return pd.concat(df_aug_list, ignore_index=True)
    
    def create_sequences(self, df, target_col, sequence_length=10, stride=1):
        """
        创建时间序列序列（用于深度学习模型）
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        target_col : str
            目标列名称
        sequence_length : int
            序列长度
        stride : int
            步长
            
        Returns:
        --------
        tuple : (X, y) 序列数据和标签
        """
        feature_cols = [col for col in df.columns if col != target_col]
        
        X, y = [], []
        for i in range(0, len(df) - sequence_length, stride):
            seq = df[feature_cols].iloc[i:i+sequence_length].values
            label = df[target_col].iloc[i+sequence_length]
            X.append(seq)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def add_trend_features(self, df, columns, window=10):
        """
        添加趋势特征
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list
            要处理的列
        window : int
            趋势计算窗口
            
        Returns:
        --------
        DataFrame : 包含趋势特征的数据
        """
        df_trend = df.copy()
        
        for col in columns:
            # 线性趋势斜率
            df_trend[f'{col}_trend'] = df_trend[col].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
            )
            # 趋势方向
            df_trend[f'{col}_trend_direction'] = np.sign(df_trend[f'{col}_trend'])
            # 变化率
            df_trend[f'{col}_change_rate'] = df_trend[col].pct_change()
            # 加速度
            df_trend[f'{col}_acceleration'] = df_trend[f'{col}_change_rate'].diff()
        
        return df_trend
    
    def add_seasonal_features(self, df, time_column='Time'):
        """
        添加季节性特征
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        time_column : str
            时间列名称
            
        Returns:
        --------
        DataFrame : 包含季节性特征的数据
        """
        df_seasonal = df.copy()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_seasonal[time_column]):
            df_seasonal[time_column] = pd.to_datetime(df_seasonal[time_column], errors='coerce')
        
        # 周期性特征（正弦/余弦编码）
        df_seasonal['hour_sin'] = np.sin(2 * np.pi * df_seasonal[time_column].dt.hour / 24)
        df_seasonal['hour_cos'] = np.cos(2 * np.pi * df_seasonal[time_column].dt.hour / 24)
        df_seasonal['day_sin'] = np.sin(2 * np.pi * df_seasonal[time_column].dt.dayofyear / 365)
        df_seasonal['day_cos'] = np.cos(2 * np.pi * df_seasonal[time_column].dt.dayofyear / 365)
        df_seasonal['week_sin'] = np.sin(2 * np.pi * df_seasonal[time_column].dt.week / 52)
        df_seasonal['week_cos'] = np.cos(2 * np.pi * df_seasonal[time_column].dt.week / 52)
        
        return df_seasonal
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42, time_series=False):
        """
        数据划分
        
        Parameters:
        -----------
        X : DataFrame or array
            特征数据
        y : DataFrame or array
            目标数据
        test_size : float
            测试集比例
        val_size : float
            验证集比例
        random_state : int
            随机种子
        time_series : bool
            是否使用时序划分（按顺序划分，适用于时序数据）
        """
        if time_series:
            X = np.asarray(X) if hasattr(X, 'values') else np.array(X)
            y = np.asarray(y) if hasattr(y, 'values') else np.array(y)
            
            n_samples = len(X)
            test_start = int(n_samples * (1 - test_size))
            val_start = int(test_start * (1 - val_size)) if val_size > 0 else test_start
            
            X_train = X[:val_start]
            X_val = X[val_start:test_start] if val_size > 0 else None
            X_test = X[test_start:]
            
            y_train = y[:val_start]
            y_val = y[val_start:test_start] if val_size > 0 else None
            y_test = y[test_start:]
            
            if val_size > 0:
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                return X_train, None, X_test, y_train, None, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            if val_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state
                )
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                return X_train, None, X_test, y_train, None, y_test
    
    def create_lagged_features(self, df, columns, lags=1):
        """
        创建滞后特征
        """
        df_lagged = df.copy()
        
        for col in columns:
            for lag in range(1, lags + 1):
                df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
        
        # 删除包含NaN的行
        df_lagged = df_lagged.dropna()
        
        return df_lagged
    
    def normalize_data(self, X):
        """
        数据归一化
        """
        if self.scaler is None:
            raise ValueError("请先调用scale_data方法")
        
        X_normalized = self.scaler.transform(X)
        return pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
    
    def inverse_scale_data(self, X_scaled):
        """
        反归一化数据
        """
        if self.scaler is None:
            raise ValueError("请先调用scale_data方法")
        
        X_original = self.scaler.inverse_transform(X_scaled)
        return pd.DataFrame(X_original, columns=X_scaled.columns, index=X_scaled.index)
    
    def rolling_window(self, df, columns, window_size=3, functions=['mean', 'std', 'min', 'max']):
        """
        滚动窗口统计特征
        """
        df_rolling = df.copy()
        
        for col in columns:
            for func in functions:
                df_rolling[f'{col}_roll{window_size}_{func}'] = df_rolling[col].rolling(window=window_size).agg(func)
        
        return df_rolling
    
    def differencing(self, df, columns, order=1):
        """
        差分操作
        """
        df_diff = df.copy()
        
        for col in columns:
            for i in range(1, order + 1):
                df_diff[f'{col}_diff{i}'] = df_diff[col].diff(i)
        
        return df_diff
    
    def extract_time_features(self, df, time_column='Time'):
        """
        从时间列提取特征
        """
        df_time = df.copy()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_time[time_column]):
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors='coerce')
        
        # 提取时间特征
        df_time[f'{time_column}_hour'] = df_time[time_column].dt.hour
        df_time[f'{time_column}_day'] = df_time[time_column].dt.day
        df_time[f'{time_column}_week'] = df_time[time_column].dt.week
        df_time[f'{time_column}_month'] = df_time[time_column].dt.month
        df_time[f'{time_column}_quarter'] = df_time[time_column].dt.quarter
        df_time[f'{time_column}_year'] = df_time[time_column].dt.year
        df_time[f'{time_column}_dayofweek'] = df_time[time_column].dt.dayofweek
        df_time[f'{time_column}_is_weekend'] = df_time[time_column].dt.dayofweek >= 5
        
        return df_time
    
    def select_features(self, X, y, k=10, method='f_regression'):
        """
        特征选择
        """
        if method == 'f_regression':
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'rfe':
            # 递归特征消除
            estimator = LinearRegression()
            self.feature_selector = RFE(estimator, n_features_to_select=k)
        elif method == 'tree_based':
            # 基于树的特征选择
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.feature_selector = SelectFromModel(estimator, max_features=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
