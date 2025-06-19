import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV 分析工具", layout="wide")
st.title("📊 上傳並分析你的 CSV 檔案")

uploaded_file = st.file_uploader("請上傳 CSV 檔案", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("✅ 上傳成功！")

    st.subheader("📋 資料預覽")
    st.dataframe(df)

    st.subheader("📈 統計摘要")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        col = st.selectbox("選擇欄位繪圖", numeric_cols)
        st.bar_chart(df[col])
    else:
        st.warning("找不到數值欄位")
