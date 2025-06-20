import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import json # 新增：用於讀取 JSON 語言檔案
# import os # 在這個版本中未直接使用os.environ，但引入它以備不時之需，保持不動

# --- Streamlit 頁面設定 (必須是第一個 Streamlit 命令，在任何其他 st. 開頭的命令之前) ---
# 設定寬版面。如果您想要內容內容居中，可以改為 layout="centered"
st.set_page_config(page_title="心理健康資料分析 + AI 問答", layout="wide")

# **注意：更改主題顏色需在專案目錄下 .streamlit/config.toml 中設定。**
# 範例 config.toml 內容：
# [theme]
# primaryColor="#4CAF50" # 綠色系按鈕和高亮
# backgroundColor="#E8F5E9" # 淺綠色背景
# secondaryBackgroundColor="#C8E6C9" # 略深一點的綠色側邊欄
# textColor="#212121"     # 深灰文字
# font="sans serif" # 字體 (可以是 'sans serif', 'serif', 'monospace')

# --- 多語言支援設定 ---
LANG_FILES = {
    "繁體中文": "zh_tw.json",
    "English": "en.json"
}
DEFAULT_LANG = "繁體中文" # 預設語言

# 在側邊欄添加語言選擇器
st.sidebar.subheader("🌍 語言選擇")
selected_lang_name = st.sidebar.selectbox("選擇語言", list(LANG_FILES.keys()), index=list(LANG_FILES.keys()).index(DEFAULT_LANG), key="lang_selector")

# 載入選定的語言檔案
@st.cache_data(show_spinner=False) # 快取語言檔案，避免每次重載
def load_language_file(lang_file_name):
    try:
        with open(f"lang/{lang_file_name}", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ 語言檔案 '{lang_file_name}' 未找到。請確保 'lang' 資料夾存在且包含正確的檔案。")
        return {} # 返回空字典以避免錯誤

translations = load_language_file(LANG_FILES[selected_lang_name])

# 定義一個簡潔的翻譯函數
def _(key, **kwargs):
    text = translations.get(key, key) # 如果找不到 key，就直接返回 key 本身
    if kwargs:
        text = text.format(**kwargs) # 格式化帶有參數的文字 (例如 {model_name})
    return text


# --- 設定 Gemini API 金鑰 ---
# 推薦使用 st.secrets 從 secrets.toml 檔案中安全地讀取金鑰
gemini_api_key = None # 初始化為 None
try:
    # 嘗試從 Streamlit Secrets 獲取 API 金鑰
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    # 如果找不到金鑰，顯示錯誤訊息和設定指南
    st.error(_("error_api_key_not_found"))
    st.markdown(_("api_key_setup_guide"))
    st.markdown(_("api_key_guide_1"))
    st.markdown(_("api_key_guide_2_local"))
    st.markdown(_("api_key_guide_2_local_a"))
    st.markdown(_("api_key_guide_2_local_b"))
    st.markdown(_("api_key_guide_2_local_c"))
    st.code("""
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
""")
    st.markdown(_("api_key_guide_3_cloud"))
    st.markdown(_("api_key_guide_3_cloud_a"))
    st.markdown(_("api_key_guide_3_cloud_b"))
    st.markdown(_("api_key_guide_3_cloud_finish"))
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
    st.sidebar.subheader(_("sidebar_gemini_status"))
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
            model = genai.GenerativeModel(TARGET_GEMINI_MODEL)
            st.sidebar.success(_("success_model_loaded", model_name=TARGET_GEMINI_MODEL))
            gemini_api_working = True
        else:
            st.sidebar.error(_("error_model_not_available", model_name=TARGET_GEMINI_MODEL))
            st.sidebar.info(_("info_check_api_or_try_other"))

    except Exception as e:
        st.sidebar.error(_("error_api_connection") + f" {e}")
        st.sidebar.info(_("info_check_network_or_api"))
else:
    # 如果 gemini_api_key 不存在 (即 Key Error 發生)，則在側邊欄顯示提示
    st.sidebar.subheader(_("sidebar_gemini_status"))
    st.sidebar.warning(_("warning_api_key_not_provided"))


# --- 頁面標題與圖片 (現在這些都會在 set_page_config 之後執行) ---
# 您可以將圖片檔案放在與 aaa.py 同一資料夾下，例如 'logo.png'
# st.image("logo.png", width=80) # 使用本地圖片範例 (需要您有 'logo.png' 檔案)
st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=80) # 使用網路圖片
st.title(_("app_title"))
st.markdown(_("app_description"))

# --- 功能切換：從側邊欄選單改為頁面頂部的 Tabs ---
tab_csv_upload, tab_gemini_ai = st.tabs([_("tab_csv_upload"), _("tab_gemini_ai")])

# --- 功能一：CSV 上傳與分析 (現在在第一個 Tab 中) ---
with tab_csv_upload:
    st.subheader(_("section_data_analysis"))
    uploaded_file = st.file_uploader(_("upload_csv_prompt"), type="csv", key="csv_uploader_main")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(_("upload_success"))
            st.dataframe(df.head()) # 顯示前幾行數據

            # 顯示統計摘要
            st.markdown("### " + _("data_overview"))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(_("metric_data_rows"), len(df))
            with col2:
                st.metric(_("metric_num_cols"), len(df.columns))
            with col3:
                st.metric(_("metric_missing_values"), df.isnull().sum().sum())

            # 使用 Expander 組織視覺化內容
            with st.expander(_("expander_numeric_viz")):
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox(_("select_numeric_col"), numeric_cols, key="numeric_col_select")

                    chart_type = st.radio(
                        _("select_chart_type"),
                        [_("chart_type_hist"), _("chart_type_line"), _("chart_type_box")],
                        horizontal=True, # 讓選項橫向排列
                        key="chart_type_radio"
                    )

                    if chart_type == _("chart_type_hist"):
                        st.write(f"**{selected_col} {_('histogram_title')}**")
                        fig, ax = plt.subplots()
                        df[selected_col].hist(ax=ax, bins=20, edgecolor='black')
                        ax.set_title(f'{selected_col} ' + _('chart_type_hist'))
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel(_('xlabel_freq'))
                        st.pyplot(fig)
                    elif chart_type == _("chart_type_line"):
                        st.write(f"**{selected_col} {_('chart_type_line')}**")
                        st.line_chart(df[selected_col])
                    elif chart_type == _("chart_type_box"):
                        st.write(f"**{selected_col} {_('chart_type_box')}**")
                        fig, ax = plt.subplots()
                        df.boxplot(column=selected_col, ax=ax)
                        ax.set_title(f'{selected_col} ' + _('chart_type_box'))
                        st.pyplot(fig)
                else:
                    st.warning(_("warning_no_numeric_cols"))

            with st.expander(_("expander_categorical_viz")):
                categorical_cols = df.select_dtypes(include='object').columns
                if len(categorical_cols) > 0:
                    selected_cat_col = st.selectbox(_("select_cat_col"), categorical_cols, key="cat_col_select")
                    st.write(f"**{selected_cat_col} {_('bar_chart_title')}**")
                    st.bar_chart(df[selected_cat_col].value_counts())
                else:
                    st.info(_("info_no_cat_cols"))

        except Exception as e:
            st.error(_("error_read_csv") + f" {e}")
            st.info(_("info_check_csv_format"))
    else:
        st.info(_("info_upload_csv"))

# --- 功能二：Gemini AI 問答 (現在在第二個 Tab 中) ---
with tab_gemini_ai:
    st.subheader(_("section_ai_assistant"))
    if not gemini_api_working: # 如果 API 或模型未成功載入，禁用 AI 功能
        st.warning(_("warning_ai_unavailable"))
        st.info(_("info_check_api_key"))
    else:
        # --- 會話歷史管理 ---
        # 初始化聊天歷史
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 重置聊天按鈕
        col_clear, col_spacer = st.columns([0.2, 0.8])
        with col_clear:
            if st.button(_("button_clear_chat"), key="clear_chat_button"):
                st.session_state.messages = []
                st.session_state.chat = None # 清除 chat object 以便重新初始化
                st.success(_("chat_clear_success"))
                st.rerun() # 重新運行應用程式以刷新介面和聊天對象

        # 創建一個聊天對象。每次重新運行應用程式時都會創建新的聊天對象
        # 但我們會將歷史記錄從 session_state 傳入
        if "chat" not in st.session_state or st.session_state.chat is None: # 確保在清空後重新創建
            try:
                st.session_state.chat = model.start_chat(history=st.session_state.messages)
            except Exception as e:
                st.error(_("error_chat_init_failed") + f" {e}")
                st.info("這可能是由於 API 金鑰問題或模型無法初始化。") # 這條資訊沒有翻譯鍵，保留原文
                st.session_state.chat = None # 確保 chat 物件為 None

        # 顯示歷史訊息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"]) # 使用 markdown 更好地顯示內容

        st.info(_("ai_prompt"))
        user_input = st.chat_input(_("chat_input_placeholder"), key="gemini_query_input") # 使用 st.chat_input 改善體驗

        if user_input:
            # 將使用者訊息添加到聊天歷史中
            st.session_state.messages.append({"role": "user", "parts": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.chat: # 只有在 chat 物件成功創建後才嘗試發送請求
                with st.spinner(_("gemini_thinking")):
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
                        st.error(_("error_gemini_communication") + f" {e}")
                        st.warning(_("warning_api_or_network_policy"))
                        st.info(_("info_check_api_and_policy"))
            else:
                st.error(_("error_chat_init_failed")) # 這裡因為 chat 對象未成功創建，所以直接報錯
                st.info(_("info_check_api_key")) # 提示用戶檢查 API 金鑰