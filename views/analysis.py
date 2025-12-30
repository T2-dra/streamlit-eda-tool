import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def show_ai_analysis(df):
    """タブ4: AIによる重要度分析を表示"""
    st.subheader("🤖 AI自動分析（要因探索）")
    st.markdown("設定した**「目的変数（予測したい列）」**に対して、どの列が重要かをAIが算出します。")

    # 1. 目的変数の選択
    target_col = st.selectbox("🎯 予測したい列（目的変数）を選んでください", df.columns)

    # 2. 分析実行ボタン
    if st.button("分析を開始する"):
        with st.spinner("AIがデータを学習中..."):
            try:
                # --- データ前処理（ここがデータサイエンスの肝です！） ---
                
                # A. データのコピー（元のdfを壊さないため）
                model_df = df.copy()
                
                # B. 欠損値の処理
                # 数値列は「平均値」、カテゴリ列は「最頻値」で埋める
                num_cols = model_df.select_dtypes(include=['number']).columns
                cat_cols = model_df.select_dtypes(exclude=['number']).columns
                
                # 数値: 平均埋め
                if len(num_cols) > 0:
                    imputer_num = SimpleImputer(strategy='mean')
                    model_df[num_cols] = imputer_num.fit_transform(model_df[num_cols])
                
                # カテゴリ: 最頻値埋め & 数値化（LabelEncoding）
                le = LabelEncoder()
                for col in cat_cols:
                    # 文字列に変換して欠損を埋める
                    model_df[col] = model_df[col].astype(str).fillna('Missing')
                    # 数値に変換（例: "東京"->0, "大阪"->1）
                    model_df[col] = le.fit_transform(model_df[col])

                # --- 学習データの作成 ---
                X = model_df.drop(columns=[target_col]) # 特徴量（原因）
                y = model_df[target_col]                # 目的変数（結果）

                # --- モデルの選択と学習 ---
                # 目的変数が数値なら「回帰」、カテゴリなら「分類」を自動選択
                if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 10:
                    model = RandomForestRegressor(n_jobs=-1, random_state=42)
                    algo_name = "ランダムフォレスト（回帰）"
                else:
                    model = RandomForestClassifier(n_jobs=-1, random_state=42)
                    algo_name = "ランダムフォレスト（分類）"

                model.fit(X, y)

                # --- 結果の可視化 ---
                st.success(f"学習完了！ 使用アルゴリズム: {algo_name}")
                
                # 重要度の取得
                importances = model.feature_importances_
                feature_names = X.columns
                
                # データフレーム化してソート
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True) # グラフ用に昇順

                # 棒グラフで表示
                fig = px.bar(
                    imp_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title=f"「{target_col}」への影響度ランキング",
                    height=max(400, len(feature_names) * 20) # 列数に応じて縦に伸ばす
                )
                st.plotly_chart(fig, width="stretch")
                
                st.info("💡 棒グラフが長いほど、予測に強く寄与している重要なデータです。")

            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")