import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def show_summary(df):
    """タブ1: データ概要を表示"""
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

def show_interactive_charts(df):
    """タブ2: インタラクティブなグラフを表示"""
    st.subheader("変数の分布と関係性（インタラクティブ）")
    
    # 修正①: ラジオボタンにもkeyを追加
    chart_type = st.radio(
        "分析モードを選択", 
        ["数値データの分布 (ヒストグラム/散布図)", "カテゴリデータの分布 (棒グラフ)"],
        horizontal=True,
        key="radio_main_chart_type"
    )

    if chart_type == "数値データの分布 (ヒストグラム/散布図)":
        num_cols = df.select_dtypes(include=['number']).columns
        # 修正②: このselectboxにもkeyを追加
        sub_chart_type = st.selectbox(
            "グラフタイプ", 
            ["ヒストグラム (1変数)", "散布図 (2変数)"],
            key="select_sub_chart_type"
        )
        
        if sub_chart_type == "ヒストグラム (1変数)":
            # 修正③: ヒストグラム用の一意なkey
            selected_col = st.selectbox("列を選択", num_cols, key="select_hist_col")
            if selected_col:
                fig = px.histogram(df, x=selected_col, nbins=30, title=f"{selected_col} の分布")
                st.plotly_chart(fig, width="stretch")
        
        elif sub_chart_type == "散布図 (2変数)":
            c1, c2, c3 = st.columns(3)
            # 修正④: 散布図用の3つのウィジェットにも全てkeyを追加
            x_col = c1.selectbox("X軸", num_cols, index=0, key="select_scatter_x")
            y_col = c2.selectbox("Y軸", num_cols, index=1 if len(num_cols)>1 else 0, key="select_scatter_y")
            color_col = c3.selectbox("色分け（カテゴリ）", [None] + list(df.columns), key="select_scatter_color")
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, width="stretch")

    elif chart_type == "カテゴリデータの分布 (棒グラフ)":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            st.info("カテゴリ列はありません。")
        else:
            # 修正⑤: 棒グラフ用の一意なkey（重要！）
            selected_cat_col = st.selectbox("列を選択", cat_cols, key="select_bar_cat_col")
            if selected_cat_col:
                count_df = df[selected_cat_col].value_counts().reset_index()
                count_df.columns = [selected_cat_col, 'count']
                
                if len(count_df) > 50:
                    st.caption(f"※上位50件のみ表示しています")
                    count_df = count_df.head(50)
                
                fig = px.bar(count_df, x='count', y=selected_cat_col, orientation='h', title=f"{selected_cat_col} の内訳")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, width="stretch")

def show_correlation(df):
    """タブ3: 相関ヒートマップを表示"""
    st.subheader("数値データの相関")
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("数値列が不足しています。")