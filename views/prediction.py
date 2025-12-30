import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def show_prediction():
    """タブ5: 未来予測機能を表示"""
    st.subheader("🔮 未来予測（推論実行）")
    st.markdown("""
    **Step 1:** 「AI自動分析」で作ったモデルファイル (`.pkl`) をアップロードします。  
    **Step 2:** 予測したい新しいデータ (`CSV`) をアップロードすると、AIが未来を予測します。
    """)

    # 1. モデルのアップロード
    st.markdown("#### 1. モデルファイルの読み込み")
    uploaded_model = st.file_uploader("学習済みモデル (.pkl) をアップロード", type="pkl", key="model_uploader")

    if uploaded_model is not None:
        try:
            # モデルデータの読み込み
            model_data = joblib.load(uploaded_model)
            model = model_data["model"]
            features = model_data["features"]
            target_name = model_data.get("target", "予測結果")
            
            st.success(f"モデルを読み込みました！ ({model_data.get('algo_name', '不明なモデル')})")
            
            # 2. 予測用データのアップロード
            st.markdown("#### 2. 予測したいデータの読み込み")
            input_file = st.file_uploader("予測用データ (CSV) をアップロード", type="csv", key="pred_data_uploader")

            if input_file is not None:
                input_df = pd.read_csv(input_file)
                
                if st.button("予測を開始する", key="btn_predict"):
                    # 列の過不足チェック
                    missing_cols = set(features) - set(input_df.columns)
                    if missing_cols:
                        st.error(f"エラー: データに以下の列が足りません。\n{missing_cols}")
                    else:
                        try:
                            # --- 予測用データの前処理 ---
                            # 学習時と同じ列順序に並べ替え
                            X_pred = input_df[features].copy()
                            
                            # 数値以外の列（文字列）を探して、数値に変換する
                            cat_cols = X_pred.select_dtypes(exclude=['number']).columns
                            
                            le = LabelEncoder()
                            for col in cat_cols:
                                # 文字列化して欠損を埋める
                                X_pred[col] = X_pred[col].astype(str).fillna('Missing')
                                # 数値に変換（例: "Male"->1, "Female"->0）
                                X_pred[col] = le.fit_transform(X_pred[col])
                            
                            # 数値列の欠損値も0で埋めておく（念のため）
                            X_pred = X_pred.fillna(0)
                            
                            # --- 予測実行 ---
                            predictions = model.predict(X_pred)
                            
                            # 結果の表示
                            result_df = input_df.copy()
                            result_df[f"予測結果_{target_name}"] = predictions
                            
                            st.write("### 🎯 予測結果")
                            st.dataframe(result_df)
                            
                            # CSVダウンロード
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "予測結果をCSVでダウンロード",
                                csv,
                                "prediction_result.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"予期せぬエラーが発生しました: {e}")

        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")