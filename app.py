import streamlit as st
import matplotlib_fontja
# 自作モジュールの読み込み
from utils.loader import load_data
from views.charts import show_summary, show_interactive_charts, show_correlation
from views.analysis import show_ai_analysis

# 1. ページの設定
st.set_page_config(page_title="EDAツール", layout="wide")
st.title("📊 EDAツール")

# 2. サイドバー：データ入力
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type="csv")

if uploaded_file is not None:
    # データ読み込み (Utils)
    df = load_data(uploaded_file, uploaded_file.name)
    
    st.sidebar.write("---")
    st.sidebar.write(f"行数: {df.shape[0]}")
    st.sidebar.write(f"列数: {df.shape[1]}")

    # 3. メイン画面の構成
    tab1, tab2, tab3, tab4 = st.tabs(["📋 データ概要", "📈 詳細グラフ", "🔥 相関ヒートマップ", "🤖 AI自動分析"])

    with tab1:
        show_summary(df)

    with tab2:
        show_interactive_charts(df)

    with tab3:
        show_correlation(df)

    with tab4:
        show_ai_analysis(df)
else:
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください。")