import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import ta
from vnstock import stock_historical_data
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings

warnings.filterwarnings('ignore')

TRAIN_PERIOD = 252;
TEST_PERIOD = 63;
STEP = 21
RF_PARAMS = {'n_estimators': 100, 'max_depth': 8, 'class_weight': 'balanced', 'random_state': 42}
FEATURES_4 = ['RSI', 'MACD_Hist', 'ADX', 'STO']
FEATURES_6 = ['RSI', 'MACD_Hist', 'ADX', 'STO', 'BB_P', 'Volume_Ratio']

st.set_page_config(page_title="DỰ ĐOÁN GIÁ CỔ PHIẾU", layout="wide")
st.title("📈 HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU SỬ DỤNG TẬP MỜ")

if 'step' not in st.session_state: st.session_state.step = 0
if 'df' not in st.session_state: st.session_state.df = None
if 'results' not in st.session_state: st.session_state.results = None


@st.cache_resource
def get_fuzzy_simulator():
    rsi, macd_hist, adx, sto = [ctrl.Antecedent(np.arange(0, 1.01, 0.01), n) for n in FEATURES_4]
    signal = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Signal')

    def make_mf(var, name):
        var[f'{name}_Low'] = fuzz.trapmf(var.universe, [0.0, 0.0, 0.30, 0.40])
        var[f'{name}_Mid'] = fuzz.trimf(var.universe, [0.30, 0.50, 0.70])
        var[f'{name}_High'] = fuzz.trapmf(var.universe, [0.60, 0.70, 1.0, 1.0])

    for v, n in zip([rsi, macd_hist, adx, sto], FEATURES_4): make_mf(v, n)
    signal['Down'] = fuzz.trapmf(signal.universe, [0.0, 0.0, 0.30, 0.45])
    signal['Neutral'] = fuzz.trimf(signal.universe, [0.40, 0.50, 0.60])
    signal['Up'] = fuzz.trapmf(signal.universe, [0.55, 0.70, 1.0, 1.0])
    rules = [
        ctrl.Rule(rsi['RSI_High'] & macd_hist['MACD_Hist_High'] & adx['ADX_High'], signal['Up']),
        ctrl.Rule(rsi['RSI_Low'] & macd_hist['MACD_Hist_Low'] & adx['ADX_High'], signal['Down']),
        ctrl.Rule(rsi['RSI_Low'] & sto['STO_Low'] & adx['ADX_Low'], signal['Up']),
        ctrl.Rule(rsi['RSI_High'] & sto['STO_High'] & adx['ADX_Low'], signal['Down']),
        ctrl.Rule(rsi['RSI_High'] & sto['STO_High'] & macd_hist['MACD_Hist_High'], signal['Up']),
        ctrl.Rule(rsi['RSI_Low'] & sto['STO_Low'] & macd_hist['MACD_Hist_Low'], signal['Down']),
        ctrl.Rule(adx['ADX_Low'] & rsi['RSI_Mid'], signal['Neutral']),
        ctrl.Rule(adx['ADX_Low'] & sto['STO_Mid'], signal['Neutral']),
        ctrl.Rule(rsi['RSI_High'] & macd_hist['MACD_Hist_Low'], signal['Neutral']),
        ctrl.Rule(rsi['RSI_Low'] & macd_hist['MACD_Hist_High'], signal['Neutral']),
        ctrl.Rule(rsi['RSI_Low'] & sto['STO_Low'], signal['Up']),
        ctrl.Rule(rsi['RSI_High'] & sto['STO_High'], signal['Down']),
        ctrl.Rule(macd_hist['MACD_Hist_High'] & adx['ADX_High'] & sto['STO_High'], signal['Up']),
        ctrl.Rule(macd_hist['MACD_Hist_Low'] & adx['ADX_High'] & sto['STO_Low'], signal['Down']),
        ctrl.Rule(rsi['RSI_Mid'] & macd_hist['MACD_Hist_High'] & adx['ADX_High'], signal['Up']),
        ctrl.Rule(rsi['RSI_Mid'] & macd_hist['MACD_Hist_Low'] & adx['ADX_High'], signal['Down']),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def predict_fuzzy(X_4scaled):
    preds, cont = [], [];
    simulator = get_fuzzy_simulator()
    for row in X_4scaled:
        try:
            simulator.input['RSI'] = row[0];
            simulator.input['MACD_Hist'] = row[1];
            simulator.input['ADX'] = row[2];
            simulator.input['STO'] = row[3]
            simulator.compute();
            val = float(simulator.output['Signal'])
        except:
            val = 0.5
        cont.append(val);
        preds.append(2 if val > 0.65 else 0 if val < 0.35 else 1)
    return np.array(preds), np.array(cont)


# --- Hàm chạy WFA ---
def run_wfa_and_evaluate(df, threshold):
    results = {
        'fuzzy': {'pred': [], 'true': [], 'date': [], 'close': []},
        'hybrid': {'pred': [], 'true': [], 'date': [], 'close': []}
    }
    n = len(df);
    start_idx = TRAIN_PERIOD

    while start_idx + TEST_PERIOD <= n:
        train_df = df.iloc[start_idx - TRAIN_PERIOD:start_idx];
        test_df = df.iloc[start_idx:start_idx + TEST_PERIOD]

        # Model 1 (Fuzzy)
        scaler4 = MinMaxScaler().fit(train_df[FEATURES_4]);
        X_test_4 = scaler4.transform(test_df[FEATURES_4])
        pred_f, cont_f = predict_fuzzy(X_test_4)

        # Model 2 (Hybrid)
        scaler6 = MinMaxScaler().fit(train_df[FEATURES_6]);
        X_train_6 = scaler6.fit_transform(train_df[FEATURES_6])
        X_test_6 = scaler6.transform(test_df[FEATURES_6])
        fuzzy_train = predict_fuzzy(scaler4.transform(train_df[FEATURES_4]))[1].reshape(-1, 1)
        X_train_h = np.hstack([X_train_6, fuzzy_train]);
        X_test_h = np.hstack([X_test_6, cont_f.reshape(-1, 1)])

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train_h, train_df['Target'])
        pred_h = rf.predict(X_test_h)

        # Lưu kết quả
        results['fuzzy']['pred'].extend(pred_f);
        results['fuzzy']['true'].extend(test_df['Target'].tolist());
        results['fuzzy']['date'].extend(test_df['time'].tolist());
        results['fuzzy']['close'].extend(test_df['close'].tolist())
        results['hybrid']['pred'].extend(pred_h);
        results['hybrid']['true'].extend(test_df['Target'].tolist());
        results['hybrid']['date'].extend(test_df['time'].tolist());
        results['hybrid']['close'].extend(test_df['close'].tolist())

        start_idx += STEP

    for m in results:
        for k in results[m]: results[m][k] = np.array(results[m][k])
    return results


# ================================================================
# ================================================================

col1, col2 = st.columns(2)
with col1:
    stock_code = st.text_input("Nhập mã cổ phiếu", value="VIC")
    threshold_pct = st.slider("Ngưỡng phân lớp (%)", 0.1, 2.0, 0.5, 0.1)
    threshold = threshold_pct / 100
with col2:
    start_date = st.date_input("Từ ngày", value=pd.to_datetime("2015-01-01"))
    end_date = st.date_input("Đến ngày", value=pd.to_datetime("2025-11-11"))

st.markdown("---")

# --- CONTROL BUTTONS ---
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button("1. TẢI DỮ LIỆU & EDA", type="primary", disabled=st.session_state.step > 0):
        with st.spinner("Bước 1: Đang tải dữ liệu và tính chỉ báo..."):
            df_raw = stock_historical_data(stock_code, str(start_date), str(end_date))
            df_raw = df_raw[["time", "open", "high", "low", "close", "volume"]].copy()
            df_raw["time"] = pd.to_datetime(df_raw["time"])
            df_raw.sort_values("time", inplace=True)

            # Tính chỉ báo
            df_raw['RSI'] = ta.momentum.RSIIndicator(df_raw['close'], 14).rsi()
            df_raw['MACD_Hist'] = ta.trend.MACD(df_raw['close']).macd_diff()
            df_raw['ADX'] = ta.trend.ADXIndicator(df_raw['high'], df_raw['low'], df_raw['close'], 14).adx()
            df_raw['STO'] = ta.momentum.StochasticOscillator(df_raw['high'], df_raw['low'], df_raw['close'], 14).stoch()
            df_raw['BB_P'] = ta.volatility.BollingerBands(df_raw['close'], 20).bollinger_pband()
            df_raw['Volume_Ratio'] = df_raw['volume'] / df_raw['volume'].rolling(20).mean()

            # Xử lý NaN & Target
            df_raw.dropna(inplace=True);
            df_raw.reset_index(drop=True, inplace=True)
            future_return = df_raw['close'].shift(-1) / df_raw['close'] - 1
            df_raw['Target'] = np.select([future_return > threshold, future_return < -threshold], [2, 0], default=1)
            df_raw = df_raw.iloc[:-1].copy();
            df_raw.reset_index(drop=True, inplace=True)

            st.session_state.df = df_raw
            st.session_state.step = 1
            st.success(f"Bước 1 hoàn thành! Số mẫu: {len(df_raw)}")

with btn_col2:
    if st.button("2. CHẠY MÔ HÌNH", disabled=st.session_state.step < 1 or st.session_state.step == 2):
        with st.spinner("Bước 2: Đang huấn luyện RF và chạy WFA..."):
            st.session_state.results = run_wfa_and_evaluate(st.session_state.df, threshold)
            st.session_state.step = 2
        st.success("Bước 2 hoàn thành! Kết quả sẵn sàng hiển thị.")

with btn_col3:
    if st.button("3. XEM KẾT QUẢ ĐÁNH GIÁ", type="secondary", disabled=st.session_state.step < 2):
        st.session_state.step = 3
        st.balloons()

st.markdown("---")

# ================================================================
# 4. KHU VỰC HIỂN THỊ KẾT QUẢ THEO TỪNG BƯỚC
# ================================================================

# --- BƯỚC 1: DỮ LIỆU & BIỂU ĐỒ CHỈ BÁO ---
if st.session_state.step >= 1:
    df = st.session_state.df
    st.markdown("### **DỮ LIỆU MẪU (10 DÒNG ĐẦU)**")
    st.dataframe(df[["time", "open", "high", "low", "close", "volume"]].head(10), use_container_width=True)

    st.markdown("### **BIỂU ĐỒ GIÁ CỔ PHIẾU THEO THỜI GIAN**")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['time'], df['close'], color='blue', linewidth=1.2)
    ax.set_title(f"Giá cổ phiếu {stock_code} theo thời gian", fontsize=16, fontweight='bold')
    ax.set_xlabel("Thời gian", fontsize=12);
    ax.set_ylabel("Giá đóng cửa (VND)", fontsize=12)
    ax.tick_params(axis='x', rotation=45);
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("### **BIỂU ĐỒ 6 CHỈ BÁO KỸ THUẬT**")
    if len(df) < 100:
        st.warning("Dữ liệu quá ít để hiển thị biểu đồ chỉ báo!")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"6 Chỉ báo kỹ thuật của {stock_code}", fontsize=16, fontweight='bold')
        plot_data = df.tail(500)
        # 1. RSI
        axes[0, 0].plot(plot_data['time'], plot_data['RSI'], color='purple', linewidth=1);
        axes[0, 0].axhline(70, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(30, color='green', linestyle='--', alpha=0.7);
        axes[0, 0].set_title("RSI (14)");
        axes[0, 0].grid(alpha=0.3)
        # 2. MACD Histogram
        axes[0, 1].bar(plot_data['time'], plot_data['MACD_Hist'], color='orange', alpha=0.7, width=1);
        axes[0, 1].axhline(0, color='black', linewidth=0.8)
        axes[0, 1].set_title("MACD Histogram");
        axes[0, 1].grid(alpha=0.3)
        # 3. ADX
        axes[0, 2].plot(plot_data['time'], plot_data['ADX'], color='brown', linewidth=1);
        axes[0, 2].axhline(25, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_title("ADX (14)");
        axes[0, 2].grid(alpha=0.3)
        # 4. Stochastic
        axes[1, 0].plot(plot_data['time'], plot_data['STO'], color='teal', linewidth=1);
        axes[1, 0].axhline(80, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(20, color='green', linestyle='--', alpha=0.7);
        axes[1, 0].set_title("Stochastic Oscillator (14)");
        axes[1, 0].grid(alpha=0.3)
        # 5. Bollinger Band %B
        axes[1, 1].plot(plot_data['time'], plot_data['BB_P'], color='magenta', linewidth=1);
        axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(0.0, color='green', linestyle='--', alpha=0.7);
        axes[1, 1].set_title("Bollinger Band %B");
        axes[1, 1].grid(alpha=0.3)
        # 6. Volume Ratio
        axes[1, 2].plot(plot_data['time'], plot_data['Volume_Ratio'], color='darkblue', linewidth=1);
        axes[1, 2].axhline(1.0, color='gray', linestyle='--', alpha=0.7)
        axes[1, 2].set_title("Volume Ratio (20 ngày)");
        axes[1, 2].grid(alpha=0.3)
        for ax in axes.flat: ax.tick_params(axis='x', rotation=45); ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

# --- BƯỚC 3: KẾT QUẢ ĐÁNH GIÁ (Chỉ hiện khi nút 3 được bấm) ---
if st.session_state.step == 3:
    results = st.session_state.results

    st.subheader("**KẾT QUẢ DỰ ĐOÁN MÔ HÌNH**")
    acc_f = accuracy_score(results['fuzzy']['true'], results['fuzzy']['pred'])
    acc_h = accuracy_score(results['hybrid']['true'], results['hybrid']['pred'])

    colR1, colR2 = st.columns(2)
    with colR1:
        st.metric("Model 1: Fuzzy Accuracy", f"{acc_f:.1%}")
    with colR2:
        st.metric("Model 2: Hybrid RF Accuracy", f"{acc_h:.1%}", f"{(acc_h - acc_f) / acc_f:+.1%}")

    # Ma trận nhầm lẫn
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, model, cmap) in enumerate([
        ("Model 1: Fuzzy", 'fuzzy', 'Blues'),
        ("Model 2: Hybrid RF", 'hybrid', 'Greens')
    ]):
        cm = confusion_matrix(results[model]['true'], results[model]['pred'], labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Giảm', 'Giữ', 'Tăng'])
        disp.plot(ax=axes[i], cmap=cmap, values_format='d')
        axes[i].set_title(name)
    st.pyplot(fig)

    # Biểu đồ 1: Giá + Vùng Dự đoán (HYBRID)
    st.markdown("### **GIÁ THỰC TẾ & VÙNG DỰ ĐOÁN (MODEL 2: HYBRID RF)**")
    fig, ax = plt.subplots(figsize=(16, 7))
    dates = pd.to_datetime(results['hybrid']['date']);
    close_prices = results['hybrid']['close'];
    pred = results['hybrid']['pred']
    ax.plot(dates, close_prices, label='Giá thực tế (Close)', color='black', linewidth=1.5)
    y_min, y_max = ax.get_ylim()
    ax.fill_between(dates, y_min, y_max, where=(pred == 2), facecolor='green', alpha=0.25, label='Dự đoán Tăng')
    ax.fill_between(dates, y_min, y_max, where=(pred == 0), facecolor='red', alpha=0.25, label='Dự đoán Giảm')
    ax.set_title("So sánh Giá thực tế và Vùng xu hướng Dự đoán (Model 2 Hybrid)", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left');
    ax.grid(True, linestyle='--', alpha=0.4);
    st.pyplot(fig)

    # Biểu đồ 2: Giá + Vùng Dự đoán (FUZZY)
    st.markdown("### **GIÁ THỰC TẾ & VÙNG DỰ ĐOÁN (MODEL 1: FUZZY)**")
    fig, ax = plt.subplots(figsize=(16, 7))
    dates_f = pd.to_datetime(results['fuzzy']['date']);
    close_f = results['fuzzy']['close'];
    pred_f = results['fuzzy']['pred']
    ax.plot(dates_f, close_f, label='Giá thực tế (Close)', color='black', linewidth=1.5)
    y_min, y_max = ax.get_ylim()
    ax.fill_between(dates_f, y_min, y_max, where=(pred_f == 2), facecolor='blue', alpha=0.2,
                    label='Dự đoán Tăng (Fuzzy)')
    ax.fill_between(dates_f, y_min, y_max, where=(pred_f == 0), facecolor='orange', alpha=0.2,
                    label='Dự đoán Giảm (Fuzzy)')
    ax.set_title("So sánh Giá thực tế và Vùng xu hướng Dự đoán (Model 1 Fuzzy)", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left');
    ax.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig)