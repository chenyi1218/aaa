# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import io

# --- Streamlit 頁面設定 (必須是第一個 Streamlit 命令) ---
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# --- 設定 Gemini 模型名稱 ---
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
        return None
    try:
        genai.configure(api_key=api_key)
        _ = list(genai.list_models())
        model_instance = genai.GenerativeModel(target_model_name)
        return model_instance
    except Exception as e:
        return None

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
        with st.spinner("⏳ 正在讀取並分析您的 CSV 檔案..."):
            try:
                @st.cache_data
                def load_csv_data(file):
                    return pd.read_csv(file)

                df = load_csv_data(uploaded_file)
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
                            ["直方圖 (分佈)", "箱形圖 (分佈與異常值)"],
                            horizontal=True,
                            key="chart_type_radio"
                        )
                        fig, ax = plt.subplots()
                        if chart_type == "直方圖 (分佈)":
                            ax.set_title(f'{selected_col} 分佈')
                            ax.set_xlabel(selected_col)
                            ax.set_ylabel('頻率')
                            df[selected_col].hist(ax=ax, bins=20, edgecolor='black')
                            st.pyplot(fig)
                        elif chart_type == "箱形圖 (分佈與異常值)":
                            ax.set_title(f'{selected_col} 箱形圖')
                            ax.set_ylabel(selected_col)
                            df.boxplot(column=selected_col, ax=ax)
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
                if 'uploaded_df' in st.session_state:
                    del st.session_state.uploaded_df
    else:
        st.info("請上傳一個 CSV 檔案來開始分析。")
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

    if gemini_api_working and "messages" in st.session_state and st.session_state.messages:
        if st.button("🗑️ 清空聊天記錄", help="點擊此按鈕將刪除所有聊天對話記錄", key="clear_chat_button"):
            st.session_state.messages = []
            if "chat" in st.session_state:
                del st.session_state.chat
            st.rerun()

    # --- 檢查 API 金鑰按鈕 ---
    if st.button("🔍 檢查 API 金鑰", key="check_api_button"):
        with st.spinner("正在檢查金鑰..."):
            try:
                genai.configure(api_key=current_api_key)
                _ = genai.list_models()
                st.success("✅ 您的 API 金鑰有效，可以正常使用！")
            except Exception as e:
                st.error(f"❌ 您的 API 金鑰無效或發生錯誤：{e}")
                st.warning("請確認您輸入的金鑰正確，或嘗試重新生成一個。")
    
    # --- 自動生成報告按鈕 ---
    if gemini_api_working and 'uploaded_df' in st.session_state and not st.session_state.uploaded_df.empty:
        if st.button("✍️ 自動生成數據報告", help="點擊此按鈕讓 AI 根據當前數據生成一份報告", key="generate_report_button"):
            with st.spinner("AI 正在分析數據並生成報告..."):
                try:
                    df_to_analyze = st.session_state.uploaded_df

                    # 創建更精簡的數據上下文
                    data_summary_text = f"""
                    資料集包含 {df_to_analyze.shape[0]} 行和 {df_to_analyze.shape[1]} 列。
                    欄位名稱和資料類型：\n{df_to_analyze.dtypes.to_string()}
                    數值欄位的統計摘要：\n{df_to_analyze.describe().to_string()}
                    """

                    # 創建自動生成報告的專用提示詞
                    report_prompt = f"""
                    你是一位專業的數據分析師。你的任務是根據以下提供的數據摘要，生成一份專業且結構化的分析報告。

                    報告必須包含以下部分：
                    ### 1. 數據概覽
                    - 簡要描述數據集的規模（行、列數）。
                    - 簡要描述各個欄位的資料類型和缺值情況。

                    ### 2. 關鍵發現
                    - 找出數據中的主要趨勢、模式或關係。
                    - 分析數值欄位的統計摘要，例如平均數、中位數、最大值、最小值等。
                    - 分析類別欄位的分佈情況，例如各個類別的數量。

                    ### 3. 潛在洞察與建議
                    - 根據你的發現，提出有價值的洞察。
                    - 針對數據中的問題（例如：缺值），提供下一步的處理建議。
                    
                    ---
                    以下是您需要分析的 CSV 資料的上下文：

                    {data_summary_text}
                    """
                    
                    response = model.generate_content(report_prompt)
                    report_text = response.text
                    st.session_state.messages.append({"role": "model", "parts": report_text})
                    st.rerun()  # 新增這一行來更新聊天框顯示

                except Exception as e:
                    st.error(f"❌ 生成報告時發生錯誤：{e}")
                    st.info("請檢查您的 API 金鑰設定，或刷新頁面後重試。")
    
    if not gemini_api_working:
        st.warning("⚠️ Gemini AI 助理目前無法使用，因為 API 金鑰無效或模型未正確載入。")
        st.info("請輸入您的 Gemini API 金鑰並嘗試刷新頁面。")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat" not in st.session_state or st.session_state.chat is None:
            try:
                st.session_state.chat = model.start_chat(history=st.session_state.messages)
            except Exception as e:
                st.error(f"❌ 無法啟動 Gemini 聊天會話：{e}")
                st.info("這可能是由於 API 金鑰問題或模型無法初始化。")
                st.session_state.chat = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"])
        
        uploaded_df_exists = 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None and not st.session_state.uploaded_df.empty
        
        if uploaded_df_exists:
            st.info("您已上傳 CSV 檔案。您可以向 AI 助理提問關於此資料的問題！")
            st.markdown(f"**當前資料集：** {uploaded_file.name} ({st.session_state.uploaded_df.shape[0]} 行, {st.session_state.uploaded_df.shape[1]} 列)")
        else:
            st.info("您可以向 AI 助理提問任何問題！(若要提問資料內容，請先上傳 CSV 檔案)")

        user_input = st.chat_input("請輸入你的問題：", key="gemini_query_input")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "parts": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            if st.session_state.chat:
                with st.spinner("Gemini 思考中... 請稍候片刻"):
                    try:
                        full_prompt = user_input
                        if uploaded_df_exists:
                            df_to_analyze = st.session_state.uploaded_df
                            buffer = io.StringIO()
                            df_to_analyze.info(buf=buffer)
                            df_info_str = buffer.getvalue()
                            df_desc_str = df_to_analyze.describe().to_markdown()
                            
                            system_prompt = f"""
                            你是一位頂尖的數據分析師和心理健康領域的專家。
                            你的任務是根據我提供的 CSV 數據和問題，進行專業、嚴謹的分析，並給出有價值的洞察與建議。
                            你的回覆必須結構清晰，先列出你將如何分析的步驟，再給出結論。
                            請不要憑空捏造數據，所有結論都必須嚴格基於提供的數據。
                            """
                            data_context = f"""
                            以下是您需要分析的 CSV 資料的上下文。
                            資料概覽 (df.info()):
                            ```
                            {df_info_str}
                            ```
                            資料統計摘要 (df.describe()):
                            ```
                            {df_desc_str}
                            ```
                            資料前5行 (df.head()):
                            ```
                            {df_to_analyze.head().to_markdown(index=False)}
                            ```
                            我的問題是：{user_input}
                            """
                            full_prompt = system_prompt + data_context
                            st.markdown("---")
                            st.info("AI 正在分析您上傳的資料並進行深度思考...")
                        
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