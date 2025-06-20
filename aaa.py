import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# ✅ 設定 Gemini API 金鑰
genai.configure(api_key="AIzaSy***************")  # ← 請保密處理

# ✅ 建立 Gemini 模型物件
model = genai.GenerativeModel("gemini-pro")

# ✅ Streamlit 頁面設定
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# ✅ 頁面標題與圖片
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=80)
st.title("🧠 心理健康資料分析平台")
st.markdown("本平台支援上傳 CSV 檔進行視覺化分析，並可使用 Gemini AI 進行問答互動。")

# ✅ 側邊欄選單
st.sidebar.title("功能選單")
menu = st.sidebar.radio("請選擇功能", ["📁 上傳 CSV", "🤖 Gemini AI 問答"])

# 📁 功能一：CSV 上傳與分析
if menu == "📁 上傳 CSV":
    uploaded_file = st.file_uploader("請上傳 CSV 檔案", type="csv", key="csv_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ 上傳成功！以下為資料內容")
        st.dataframe(df)

        # 顯示統計摘要
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("資料筆數", len(df))
        with col2:
            st.metric("欄位數", len(df.columns))
        with col3:
            st.metric("缺值總數", df.isnull().sum().sum())

        st.markdown("### 📈 數值欄位視覺化")
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("選擇欄位", numeric_cols)
            chart_type = st.radio("選擇圖表類型", ["長條圖", "折線圖"])
            if chart_type == "長條圖":
                st.bar_chart(df[selected_col])
            else:
                st.line_chart(df[selected_col])
        else:
            st.warning("此資料集無可視覺化的數值欄位")

# 🤖 功能二：Gemini AI 問答
elif menu == "🤖 Gemini AI 問答":
    st.subheader("🤖 AI 問答助理（Gemini）")
    user_input = st.text_area("請輸入你的問題")

    if st.button("送出問題"):
        if user_input.strip() != "":
            with st.spinner("Gemini 思考中..."):
                try:
                    response = model.generate_content(user_input)
                    st.markdown("### ✨ Gemini 回覆：")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"❌ 錯誤：{e}")
        else:
            st.warning("請先輸入問題再送出。")
