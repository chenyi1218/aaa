# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import time # 用於模擬加載時間
import io # 用於將DataFrame轉換為字符串

# --- Streamlit 頁面設定 (必須是第一個 Streamlit 命令，在任何其他 st. 開頭的命令之前) ---
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# --- 設定 Gemini API 金鑰 ---
gemini_api_working = False # 標誌位，指示 Gemini API 是否可用

# 設定您想要使用的 Gemini 模型名稱
TARGET_GEMINI_MODEL = "models/gemini-1.5-flash"

# ====================================================================================
# 使用 @st.cache_resource 來快取 Gemini 模型物件的載入
@st.cache_resource
def get_gemini_model_cached(target_model_name, api_key):
    """
    快取 Gemini 模型物件的初始化。
    只有在第一次調用時會執行 genai.GenerativeModel()。
    """
    if not api_key:
        return None # 如果沒有金鑰，則無法初始化模型

    try:
        genai.configure(api_key=api_key) # 使用用戶提供的金鑰配置
        # 嘗試一個小的互動來確認模型是否真的可用，例如列出模型
        _ = list(genai.list_models()) # 嘗試列出模型以確認API連接
        model_instance = genai.GenerativeModel(target_model_name)
        return model_instance
    except Exception as e:
        # st.error 訊息在調用處統一處理，這裡只返回 None
        return None # 返回 None 表示模型載入失敗

# ====================================================================================

# --- 頁面標題與圖片 ---
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=80)
st.title("🧠 心理健康資料分析平台")
st.markdown("本平台支援上傳 CSV 檔進行視覺化分析，並可使用 Gemini AI 進行問答互動。")

# --- 功能切換：頁面頂部的 Tabs ---
tab_csv_upload, tab_gemini_ai = st.tabs(["📁 上傳 CSV", "🤖 Gemini AI 問答"])

# --- 功能一：CSV 上傳與分析 (在第一個 Tab 中) ---
with tab_csv_upload:
    st.subheader("📊 資料分析與視覺化")
    uploaded_file = st.file_uploader("請上傳 CSV 檔案", type="csv", key="csv_uploader_main")

    if uploaded_file:
        try:
            @st.cache_data
            def load_csv_data(file):
                return pd.read_csv(file)

            df = load_csv_data(uploaded_file)
            # 將 DataFrame 儲存到 session_state，供 AI 問答使用
            st.session_state.uploaded_df = df
            st.success("✅ 上傳成功！以下為資料內容預覽：")
            st.dataframe(df.head())

            st.markdown("### 📝 資料概覽")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("資料筆數", len(df))
            with col2:
                st.metric("欄位數", len(df.columns))
            with col3:
                st.metric("缺值總數", df.isnull().sum().sum())

            with st.expander("📈 點擊查看數值欄位視覺化"):
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("選擇要分析的數值欄位", numeric_cols, key="numeric_col_select")

                    chart_type = st.radio(
                        "選擇圖表類型",
                        ["直方圖 (分佈)", "折線圖 (趨勢)", "箱形圖 (分佈與異常值)"],
                        horizontal=True,
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
            # 如果上傳失敗，清空或重置 session_state 中的 DataFrame
            if 'uploaded_df' in st.session_state:
                del st.session_state.uploaded_df
    else:
        st.info("請上傳一個 CSV 檔案來開始分析。")
        # 如果沒有上傳檔案，確保 session_state 中的 DataFrame 是空的
        if 'uploaded_df' in st.session_state:
            del st.session_state.uploaded_df

# --- 功能二：Gemini AI 問答 (在第二個 Tab 中) ---
with tab_gemini_ai:
    st.subheader("🤖 AI 問答助理（Gemini）")

    # --- 用戶輸入 API 金鑰 ---
    st.markdown("#### 🔑 輸入您的 Gemini API 金鑰")
    if "gemini_api_key_input" not in st.session_state:
        st.session_state.gemini_api_key_input = ""

    current_api_key = st.text_input(
        "請在此處輸入您的 Gemini API 金鑰：",
        type="password",
        value=st.session_state.gemini_api_key_input,
        key="gemini_api_key_text_input",
        help="前往 Google AI Studio (aistudio.google.com/app/apikey) 獲取您的金鑰。"
    )

    if current_api_key and current_api_key != st.session_state.gemini_api_key_input:
        st.session_state.gemini_api_key_input = current_api_key
        st.rerun()

    # 獲取模型實例，這次會依賴用戶輸入的 current_api_key
    model = get_gemini_model_cached(TARGET_GEMINI_MODEL, current_api_key)

    if model:
        st.sidebar.success(f"✅ Gemini 模型 '{TARGET_GEMINI_MODEL}' 已成功載入。")
        gemini_api_working = True
    else:
        st.sidebar.error(f"❌ 無法載入 Gemini 模型 '{TARGET_GEMINI_MODEL}'。")
        if not current_api_key:
            st.sidebar.warning("請在 AI 問答區域輸入有效的 Gemini API 金鑰。")
        else:
            st.sidebar.info("請檢查您的 API 金鑰是否有效、網路連線，或嘗試刷新頁面。")
        gemini_api_working = False

    # --- 新增的刪除聊天記錄按鈕 ---
    if gemini_api_working and "messages" in st.session_state and st.session_state.messages:
        if st.button("🗑️ 清空聊天記錄", help="點擊此按鈕將刪除所有聊天對話記錄", key="clear_chat_button"):
            st.session_state.messages = []
            st.session_state.chat = None
            st.rerun()

    if not gemini_api_working:
        st.warning("⚠️ Gemini AI 助理目前無法使用，因為 API 金鑰無效或模型未正確載入。")
        st.info("請輸入您的 Gemini API 金鑰並嘗試刷新頁面。")
    else:
        # --- 會話歷史管理 ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat" not in st.session_state or st.session_state.chat is None:
            try:
                if model:
                    st.session_state.chat = model.start_chat(history=st.session_state.messages)
                else:
                    st.error("❌ Gemini 模型未成功載入，無法啟動聊天會話。")
                    st.session_state.chat = None
            except Exception as e:
                st.error(f"❌ 無法啟動 Gemini 聊天會話：{e}")
                st.info("這可能是由於 API 金鑰問題或模型無法初始化。")
                st.session_state.chat = None

        # 顯示歷史訊息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"])

        # 判斷是否有上傳的 CSV 資料
        uploaded_df_exists = 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None and not st.session_state.uploaded_df.empty

        if uploaded_df_exists:
            st.info("您已上傳 CSV 檔案。您可以向 AI 助理提問關於此資料的問題！")
            st.markdown(f"**當前資料集：** {uploaded_file.name} ({st.session_state.uploaded_df.shape[0]} 行, {st.session_state.uploaded_df.shape[1]} 列)")
        else:
            st.info("您可以向 AI 助理提問任何問題！(若要提問資料內容，請先上傳 CSV 檔案)")


        user_input = st.chat_input("請輸入你的問題：", key="gemini_query_input")

        if user_input:
            # 將使用者訊息添加到聊天歷史中
            st.session_state.messages.append({"role": "user", "parts": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.chat:
                with st.spinner("Gemini 思考中... 請稍候片刻"):
                    try:
                        # --- 關鍵修改：準備資料上下文並添加到提示詞中 ---
                        full_prompt = user_input
                        if uploaded_df_exists:
                            df_to_analyze = st.session_state.uploaded_df

                            # 創建資料的文字描述，包括列名、數據類型和前幾行
                            # 使用 StringIO 來捕獲 df.info() 的輸出
                            buffer = io.StringIO()
                            df_to_analyze.info(buf=buffer)
                            df_info_str = buffer.getvalue()

                            data_context = f"""
                            以下是您需要分析的 CSV 資料的上下文。請根據這些資料來回答我的問題。
                            資料概覽 (df.info()):
                            ```
                            {df_info_str}
                            ```
                            資料前5行 (df.head()):
                            ```
                            {df_to_analyze.head().to_markdown(index=False)}
                            ```
                            我的問題是：{user_input}
                            """
                            full_prompt = data_context
                            st.markdown("---") # 分隔線，讓用戶知道 AI 正在處理資料
                            st.info("AI 正在分析您上傳的資料...")


                        response = st.session_state.chat.send_message(full_prompt)

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
                st.info("請在上方輸入您的 API 金鑰，並確認模型狀態。")