import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os # 雖然在這個版本中未直接使用os.environ，但引入它以備不時之需

# --- 設定 Gemini API 金鑰 ---
# 推薦使用 st.secrets 從 .streamlit/secrets.toml 檔案中安全地讀取金鑰
try:
    # 嘗試從 Streamlit Secrets 獲取 API 金鑰
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("❌ 錯誤：找不到 Gemini API 金鑰！")
    st.markdown("""
        為了讓 AI 問答助理正常運作，請依照以下步驟設定您的 Gemini API 金鑰：

        1.  **獲取金鑰：** 前往 Google AI Studio ([aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) 獲取您的 Gemini API 金鑰。
        2.  **本地運行：**
            * 在您的 Streamlit 專案資料夾中，建立一個名為 **`.streamlit`** 的子資料夾。
            * 在 `.streamlit` 資料夾中，建立一個名為 **`secrets.toml`** 的檔案。
            * 打開 `secrets.toml`，並加入以下內容 (請將 `YOUR_ACTUAL_GEMINI_API_KEY` 替換成您的真實金鑰)：
                ```toml
                # .streamlit/secrets.toml
                GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
                ```
        3.  **部署到 Streamlit Cloud：**
            * 在 Streamlit Cloud 的應用程式設定中，找到 **"Secrets"** 或 **"Environment variables"**。
            * 設定一個新的 Secret，變數名稱為 `GEMINI_API_KEY`，值為您的完整 API 金鑰。

        完成上述步驟後，請重新運行您的應用程式。
    """)
    st.stop() # 如果沒有金鑰，則停止應用程式執行，避免後續錯誤

# --- 建立 Gemini 模型物件 ---
model = genai.GenerativeModel("gemini-pro")

# --- Streamlit 頁面設定 ---
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# --- 頁面標題與圖片 ---
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=80)
st.title("🧠 心理健康資料分析平台")
st.markdown("本平台支援上傳 CSV 檔進行視覺化分析，並可使用 Gemini AI 進行問答互動。")

# --- 側邊欄選單 ---
st.sidebar.title("功能選單")
menu = st.sidebar.radio("請選擇功能", ["📁 上傳 CSV", "🤖 Gemini AI 問答"])

# --- 功能一：CSV 上傳與分析 ---
if menu == "📁 上傳 CSV":
    st.subheader("📊 資料分析與視覺化")
    uploaded_file = st.file_uploader("請上傳 CSV 檔案", type="csv", key="csv_upload")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ 上傳成功！以下為資料內容預覽：")
            st.dataframe(df.head()) # 顯示前幾行數據

            # 顯示統計摘要
            st.markdown("### 📝 資料概覽")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("資料筆數", len(df))
            with col2:
                st.metric("欄位數", len(df.columns))
            with col3:
                st.metric("缺值總數", df.isnull().sum().sum())

            # 顯示數值欄位視覺化
            st.markdown("### 📈 數值欄位視覺化")
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("選擇要分析的數值欄位", numeric_cols)
                
                chart_type = st.radio("選擇圖表類型", ["直方圖 (分佈)", "折線圖 (趨勢)", "箱形圖 (分佈與異常值)"])
                
                if chart_type == "直方圖 (分佈)":
                    st.write(f"**{selected_col} 的直方圖：**")
                    fig, ax = plt.subplots()
                    df[selected_col].hist(ax=ax, bins=20, edgecolor='black')
                    ax.set_title(f'{selected_col} 分佈')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('頻率')
                    st.pyplot(fig)
                elif chart_type == "折線圖 (趨勢)":
                    st.write(f"**{selected_col} 的折線圖：**")
                    st.line_chart(df[selected_col])
                elif chart_type == "箱形圖 (分佈與異常值)":
                    st.write(f"**{selected_col} 的箱形圖：**")
                    fig, ax = plt.subplots()
                    df.boxplot(column=selected_col, ax=ax)
                    ax.set_title(f'{selected_col} 箱形圖')
                    st.pyplot(fig)
            else:
                st.warning("此資料集無可視覺化的數值欄位。")
            
            # 顯示類別欄位分佈
            st.markdown("### 📊 類別欄位分佈")
            categorical_cols = df.select_dtypes(include='object').columns
            if len(categorical_cols) > 0:
                selected_cat_col = st.selectbox("選擇要分析的類別欄位", categorical_cols)
                st.write(f"**{selected_cat_col} 的計數分佈：**")
                st.bar_chart(df[selected_cat_col].value_counts())
            else:
                st.info("此資料集無類別欄位可供分析。")

        except Exception as e:
            st.error(f"❌ 讀取 CSV 檔案時發生錯誤：{e}")
            st.info("請確認您上傳的是有效的 CSV 檔案，並且編碼正確。")
    else:
        st.info("請上傳一個 CSV 檔案來開始分析。")

# --- 功能二：Gemini AI 問答 ---
elif menu == "🤖 Gemini AI 問答":
    st.subheader("🤖 AI 問答助理（Gemini）")
    st.info("您可以向 AI 助理提問任何問題！")
    user_input = st.text_area("請輸入你的問題：", height=150, key="gemini_query")

    if st.button("送出問題", key="submit_gemini_query"):
        if user_input.strip() != "":
            with st.spinner("Gemini 思考中... 請稍候片刻"):
                try:
                    response = model.generate_content(user_input)
                    if response and hasattr(response, 'text'):
                        st.markdown("### ✨ Gemini 回覆：")
                        st.write(response.text)
                    else:
                        st.error("❌ Gemini 無法生成有效回覆。")
                        st.info("可能是內容政策違規或模型內部錯誤。請嘗試更換問題。")
                except Exception as e:
                    st.error(f"❌ 發生錯誤，無法與 Gemini 進行通訊：{e}")
                    st.warning("這可能是因為 API 金鑰無效、網路問題或請求內容不符合政策。")
                    st.info("請檢查您的 API 金鑰設定，並確保您的問題符合 Google 的使用規範。")
        else:
            st.warning("請先輸入問題再送出喔！")