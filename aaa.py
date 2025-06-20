import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
# import os # 在這個版本中未直接使用os.environ，但引入它以備不時之需，保持不動

# --- Streamlit 頁面設定 (必須是第一個 Streamlit 命令，在任何其他 st. 開頭的命令之前) ---
# 設定寬版面。如果您想要內容居中，可以改為 layout="centered"
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# **注意：更改主題顏色需在專案目錄下 .streamlit/config.toml 中設定。**
# 範例 config.toml 內容：
# [theme]
# primaryColor="#4CAF50" # 綠色系按鈕和高亮
# backgroundColor="#E8F5E9" # 淺綠色背景
# secondaryBackgroundColor="#C8E6C9" # 略深一點的綠色側邊欄
# textColor="#212121"     # 深灰文字
# font="sans serif" # 字體 (可以是 'sans serif', 'serif', 'monospace')


# --- 設定 Gemini API 金鑰 ---
# 推薦使用 st.secrets 從 secrets.toml 檔案中安全地讀取金鑰
gemini_api_key = None # 初始化為 None
try:
    # 嘗試從 Streamlit Secrets 獲取 API 金鑰
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    # 如果找不到金鑰，顯示錯誤訊息和設定指南
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
    # 不使用 st.stop()，讓使用者可以看到整個應用程式介面，但 AI 功能會被禁用

# ✅ 建立 Gemini 模型物件
model = None # 初始化為 None
gemini_api_working = False # 標誌位，指示 Gemini API 是否可用

# 設定您想要使用的 Gemini 模型名稱
# 請根據您的需求選擇以下其中一個。如果您不確定，先嘗試 gemini-pro
# TARGET_GEMINI_MODEL = "models/gemini-pro"
# TARGET_GEMINI_MODEL = "models/gemini-1.5-pro"
TARGET_GEMINI_MODEL = "models/gemini-1.5-flash"


if gemini_api_key: # 只有在有金鑰的情況下才嘗試配置和列出模型
    st.sidebar.subheader("Gemini 模型狀態")
    try:
        # 列出所有可用的模型及其支援的方法
        models_list = genai.list_models()

        # 檢查 TARGET_GEMINI_MODEL 是否可用且支援 generateContent
        target_model_available = False
        for m in models_list:
            if m.name == TARGET_GEMINI_MODEL and "generateContent" in m.supported_generation_methods:
                target_model_available = True
                break

        if target_model_available:
            # 直接使用完整且正確的模型名稱
            # 這裡我們使用 start_chat() 而不是直接 GenerativeModel()，以便更好地管理會話歷史
            model = genai.GenerativeModel(TARGET_GEMINI_MODEL)
            st.sidebar.success(f"✅ Gemini 模型 '{TARGET_GEMINI_MODEL}' 已成功載入。")
            gemini_api_working = True
        else:
            st.sidebar.error(f"❌ 模型 '{TARGET_GEMINI_MODEL}' 不可用或不支持 generateContent。")
            st.sidebar.info("請檢查您的 API 金鑰權限、地區限制或嘗試其他模型。")

    except Exception as e:
        st.sidebar.error(f"❌ 無法連接 Gemini API 或列出模型：{e}")
        st.sidebar.info("請檢查您的網路連接或 API 金鑰是否有效。")
else:
    # 如果 gemini_api_key 不存在 (即 Key Error 發生)，則在側邊欄顯示提示
    st.sidebar.subheader("Gemini 模型狀態")
    st.sidebar.warning("API 金鑰未提供，AI 問答功能將無法使用。")


# --- 頁面標題與圖片 (現在這些都會在 set_page_config 之後執行) ---
# 您可以將圖片檔案放在與 aaa.py 同一資料夾下，例如 'logo.png'
# st.image("logo.png", width=80) # 使用本地圖片範例 (需要您有 'logo.png' 檔案)
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=80) # 使用網路圖片
st.title("🧠 心理健康資料分析平台")
st.markdown("本平台支援上傳 CSV 檔進行視覺化分析，並可使用 Gemini AI 進行問答互動。")

# --- 功能切換：從側邊欄選單改為頁面頂部的 Tabs ---
tab_csv_upload, tab_gemini_ai = st.tabs(["📁 上傳 CSV", "🤖 Gemini AI 問答"])

# --- 功能一：CSV 上傳與分析 (現在在第一個 Tab 中) ---
with tab_csv_upload:
    st.subheader("📊 資料分析與視覺化")
    uploaded_file = st.file_uploader("請上傳 CSV 檔案", type="csv", key="csv_uploader_main")

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

            # 使用 Expander 組織視覺化內容
            with st.expander("📈 點擊查看數值欄位視覺化"):
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("選擇要分析的數值欄位", numeric_cols, key="numeric_col_select")

                    chart_type = st.radio(
                        "選擇圖表類型",
                        ["直方圖 (分佈)", "折線圖 (趨勢)", "箱形圖 (分佈與異常值)"],
                        horizontal=True, # 讓選項橫向排列
                        key="chart_type_radio"
                    )

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

            with st.expander("📊 點擊查看類別欄位分佈"):
                categorical_cols = df.select_dtypes(include='object').columns
                if len(categorical_cols) > 0:
                    selected_cat_col = st.selectbox("選擇要分析的類別欄位", categorical_cols, key="cat_col_select")
                    st.write(f"**{selected_cat_col} 的計數分佈：**")
                    st.bar_chart(df[selected_cat_col].value_counts())
                else:
                    st.info("此資料集無類別欄位可供分析。")

        except Exception as e:
            st.error(f"❌ 讀取 CSV 檔案時發生錯誤：{e}")
            st.info("請確認您上傳的是有效的 CSV 檔案，並且編碼正確。")
    else:
        st.info("請上傳一個 CSV 檔案來開始分析。")

# --- 功能二：Gemini AI 問答 (現在在第二個 Tab 中) ---
with tab_gemini_ai:
    st.subheader("🤖 AI 問答助理（Gemini）")

    # --- 新增的刪除聊天記錄按鈕 ---
    # 只有在 AI 助理工作時，並且有歷史訊息時才顯示刪除按鈕
    if gemini_api_working and "messages" in st.session_state and st.session_state.messages:
        if st.button("🗑️ 清空聊天記錄", help="點擊此按鈕將刪除所有聊天對話記錄"):
            st.session_state.messages = [] # 清空聊天記錄列表
            st.session_state.chat = None   # 清空聊天會話，讓其重新初始化
            st.rerun() # 重新運行應用程式，刷新顯示

    if not gemini_api_working: # 如果 API 或模型未成功載入，禁用 AI 功能
        st.warning("⚠️ Gemini AI 助理目前無法使用，因為 API 金鑰無效或模型未正確載入。")
        st.info("請檢查側邊欄的 Gemini 模型狀態和上方 API 金鑰設定說明。")
    else:
        # --- 會話歷史管理 ---
        # 初始化聊天歷史
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 創建一個聊天對象。每次重新運行應用程式時都會創建新的聊天對象
        # 但我們會將歷史記錄從 session_state 傳入
        if "chat" not in st.session_state:
            try:
                st.session_state.chat = model.start_chat(history=st.session_state.messages)
            except Exception as e:
                st.error(f"❌ 無法啟動 Gemini 聊天會話：{e}")
                st.info("這可能是由於 API 金鑰問題或模型無法初始化。")
                st.session_state.chat = None # 確保 chat 物件為 None

        # 顯示歷史訊息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"]) # 使用 markdown 更好地顯示內容

        st.info("您可以向 AI 助理提問任何問題！")
        user_input = st.chat_input("請輸入你的問題：", key="gemini_query_input") # 使用 st.chat_input 改善體驗

        if user_input:
            # 將使用者訊息添加到聊天歷史中
            st.session_state.messages.append({"role": "user", "parts": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.chat: # 只有在 chat 物件成功創建後才嘗試發送請求
                with st.spinner("Gemini 思考中... 請稍候片刻"):
                    try:
                        # 向 Gemini 發送包含完整歷史的請求
                        # 注意：genai.GenerativeModel.generate_content() 和 ChatSession.send_message() 的用法不同
                        # 對於 ChatSession，直接使用 send_message 即可，它會自動管理歷史
                        response = st.session_state.chat.send_message(user_input)

                        # 將 AI 回覆添加到聊天歷史中
                        ai_response_text = response.text
                        st.session_state.messages.append({"role": "model", "parts": ai_response_text})

                        with st.chat_message("model"):
                            st.markdown(ai_response_text)
                    except Exception as e:
                        st.error(f"❌ 發生錯誤，無法與 Gemini 進行通訊：{e}")
                        st.warning("這可能是因為 API 金鑰無效、網路問題或請求內容不符合政策。")
                        st.info("請檢查您的 API 金鑰設定，並確保您的問題符合 Google 的使用規範。")
            else:
                st.error("❌ 聊天會話未成功初始化，無法發送訊息。")
                st.info("請檢查側邊欄的 Gemini 模型狀態。")