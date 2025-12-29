import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# 1. ページの設定（ワイドモードにする）
st.set_page_config(page_title="爆速EDAツール", layout="wide")

st.title("📊 爆速EDAツール for Kaggle")

# 2. サイドバー：設定エリア
st.sidebar.header("📁 データ入力")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type="csv")

# データがある場合のみ処理を実行
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # サイドバーに基本情報を表示
    st.sidebar.write("---")
    st.sidebar.write(f"行数: {df.shape[0]}")
    st.sidebar.write(f"列数: {df.shape[1]}")

    # 3. タブの作成（3つのエリアに分ける）
    tab1, tab2, tab3 = st.tabs(["📋 データ概要", "📈 詳細グラフ", "🔥 相関ヒートマップ"])

    # --- タブ1: データ概要 ---
    with tab1:
        st.subheader("データのプレビュー")
        st.dataframe(df.head())

        st.subheader("基本統計量と欠損値")
        col1, col2 = st.columns(2)
        with col1:
            st.write("欠損値の数")
            st.dataframe(df.isnull().sum())
        with col2:
            st.write("基本統計量")
            st.dataframe(df.describe())

# --- タブ2: 詳細グラフ（インタラクティブ版） ---
    with tab2:
        st.subheader("変数の分布と関係性（インタラクティブ）")
        
        chart_type = st.radio(
            "分析モードを選択", 
            ["数値データの分布 (ヒストグラム/散布図)", "カテゴリデータの分布 (棒グラフ)"],
            horizontal=True
        )

        if chart_type == "数値データの分布 (ヒストグラム/散布図)":
            num_cols = df.select_dtypes(include=['number']).columns
            sub_chart_type = st.selectbox("グラフタイプ", ["ヒストグラム (1変数)", "散布図 (2変数)"])
            
            if sub_chart_type == "ヒストグラム (1変数)":
                selected_col = st.selectbox("列を選択", num_cols)
                if selected_col:
                    # Plotlyでヒストグラムを描画
                    fig = px.histogram(df, x=selected_col, nbins=30, title=f"{selected_col} の分布")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif sub_chart_type == "散布図 (2変数)":
                c1, c2, c3 = st.columns(3)
                x_col = c1.selectbox("X軸", num_cols, index=0)
                y_col = c2.selectbox("Y軸", num_cols, index=1 if len(num_cols)>1 else 0)
                # 色分け（カテゴリ）機能を追加
                color_col = c3.selectbox("色分け（カテゴリ）", [None] + list(df.columns))
                
                if x_col and y_col:
                    # Plotlyで散布図を描画
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "カテゴリデータの分布 (棒グラフ)":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) == 0:
                st.info("カテゴリ列はありません。")
            else:
                selected_cat_col = st.selectbox("列を選択", cat_cols)
                if selected_cat_col:
                    count_df = df[selected_cat_col].value_counts().reset_index()
                    # カラム名を修正（Plotly用に扱いやすくする）
                    count_df.columns = [selected_cat_col, 'count']
                    
                    # Top 50 制限
                    if len(count_df) > 50:
                        st.caption(f"※上位50件のみ表示しています")
                        count_df = count_df.head(50)
                    
                    # Plotlyで棒グラフを描画
                    fig = px.bar(count_df, x='count', y=selected_cat_col, orientation='h', title=f"{selected_cat_col} の内訳")
                    # 上から順に並ぶようにY軸を反転
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

    # --- タブ3: 相関ヒートマップ ---
    with tab3:
        st.subheader("数値データの相関")
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        else:
            st.info("数値列が不足しています。")

else:
    # ファイル未アップロード時の案内
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください。")