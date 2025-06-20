import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Set page config for wider layout and title
st.set_page_config(layout="wide", page_title="Mental Health Data Analysis + AI Chat")

# Load API Key from Streamlit Secrets
# For local development, ensure .streamlit/secrets.toml has GEMINI_API_KEY
# For Streamlit Cloud deployment, set GEMINI_API_KEY in Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# Target Gemini Model
# Ensure your API key has access to this model. gemini-1.5-flash is generally recommended.
# If you face issues, you might try "models/gemini-pro"
TARGET_GEMINI_MODEL = "models/gemini-1.5-flash"

# --- Language Setup ---
LANG_FILES = {
    "繁體中文": "lang/zh_tw.json",
    "English": "lang/en.json"
}

@st.cache_data # Use st.cache_data for functions that return data
def load_language_file(file_path):
    """Loads translations from a JSON file."""
    try:
        # Explicitly open with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ 語言檔案 '{file_path}' 未找到。請確保 'lang' 資料夾存在且包含正確的檔案。")
        st.stop() # Stop the app if language file is missing
    except json.JSONDecodeError as e:
        st.error(f"❌ 解析語言檔案 '{file_path}' 時發生錯誤: {e}. 請檢查 JSON 格式是否正確。")
        st.stop() # Stop the app if JSON is malformed
    except Exception as e:
        st.error(f"❌ 載入語言檔案 '{file_path}' 時發生未知錯誤: {e}")
        st.stop()

# Initialize session state for language selection
if "selected_lang_name" not in st.session_state:
    st.session_state.selected_lang_name = "繁體中文" # Default language

# Sidebar for language selection
st.sidebar.header("🌍 語言選擇")
selected_lang_name = st.sidebar.selectbox(
    "選擇語言 (Select Language)",
    list(LANG_FILES.keys()),
    key="lang_selector"
)

if selected_lang_name != st.session_state.selected_lang_name:
    st.session_state.selected_lang_name = selected_lang_name
    st.rerun() # Rerun to apply language change

translations = load_language_file(LANG_FILES[st.session_state.selected_lang_name])

# Translation function
def _(key):
    """Translates a given key using the loaded translations."""
    return translations.get(key, f"MISSING_KEY: {key}")

# --- Initialize Gemini AI ---
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
    st.session_state.chat_history = []
    st.session_state.gemini_status = _("warning_ai_unavailable") # Default status

@st.cache_resource # Use st.cache_resource for models/connections
def initialize_gemini_model(api_key, model_name):
    """Initializes the Gemini model and chat session."""
    if not api_key:
        st.session_state.gemini_status = _("warning_api_key_not_provided")
        return None

    try:
        genai.configure(api_key=api_key)
        # Attempt to list models to verify API key and connectivity
        list(genai.list_models()) # This will raise an exception if key is bad or no connection

        model = genai.GenerativeModel(model_name)
        st.session_state.gemini_status = _("success_model_loaded").format(model_name=model_name)
        return model
    except genai.types.model_types.ModelNotAvailableException:
        st.session_state.gemini_status = _("error_model_not_available").format(model_name=model_name)
        st.session_state.gemini_status += " " + _("info_check_api_or_try_other")
        return None
    except Exception as e:
        st.session_state.gemini_status = _("error_api_connection") + f" {e}"
        st.session_state.gemini_status += " " + _("info_check_network_or_api")
        return None

# Initialize model only once
if st.session_state.gemini_model is None:
    st.session_state.gemini_model = initialize_gemini_model(GEMINI_API_KEY, TARGET_GEMINI_MODEL)
    if st.session_state.gemini_model is None and GEMINI_API_KEY is not None:
        # If API key is provided but model initialization failed, warn user
        st.sidebar.warning(st.session_state.gemini_status)
    elif GEMINI_API_KEY is None:
        st.sidebar.error(_("error_api_key_not_found"))
        st.sidebar.info(_("api_key_setup_guide"))
        st.sidebar.markdown(_("api_key_guide_1"))
        st.sidebar.markdown(_("api_key_guide_2_local"))
        st.sidebar.code(_("api_key_guide_2_local_a") + "\n.streamlit/secrets.toml\n" + _("api_key_guide_2_local_b") + "\n" + _("api_key_guide_2_local_c") + "\n[secrets]\nGEMINI_API_KEY = \"YOUR_ACTUAL_GEMINI_API_KEY\"")
        st.sidebar.markdown(_("api_key_guide_3_cloud"))
        st.sidebar.markdown(_("api_key_guide_3_cloud_a") + "\n" + _("api_key_guide_3_cloud_b"))
        st.sidebar.markdown(_("api_key_guide_finish"))


# --- App Title and Description ---
st.title(_("app_title"))
st.write(_("app_description"))

# --- Tabs for Navigation ---
tab_csv_upload, tab_gemini_ai = st.tabs([_("tab_csv_upload"), _("tab_gemini_ai")])

# --- CSV Upload and Analysis Tab ---
with tab_csv_upload:
    st.header(_("section_data_analysis"))
    uploaded_file = st.file_uploader(_("upload_csv_prompt"), type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV file directly from BytesIO
            df = pd.read_csv(BytesIO(uploaded_file.getvalue()), encoding='utf-8')
            st.success(_("upload_success"))

            # --- Newly Added: Display df.head() and column list ---
            st.subheader(_("data_overview")) # General overview title
            st.write(df.head()) # Display the first few rows of the DataFrame
            st.write(_("metric_data_rows") + ":", df.shape[0]) # Display number of rows
            st.write(_("metric_num_cols") + ":", df.shape[1]) # Display number of columns
            st.write(_("column_names") + ":", df.columns.tolist()) # Display list of column names
            st.write(_("metric_missing_values") + ":", df.isnull().sum().sum()) # Display total missing values
            # --- End of newly added content ---


            # Numeric Column Visualization
            st.markdown("---")
            with st.expander(_("expander_numeric_viz")):
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_numeric_col = st.selectbox(
                        _("select_numeric_col"),
                        numeric_cols,
                        key="numeric_col_select"
                    )
                    selected_chart_type = st.selectbox(
                        _("select_chart_type"),
                        [_("chart_type_hist"), _("chart_type_line"), _("chart_type_box")],
                        key="chart_type_select"
                    )

                    if selected_numeric_col:
                        plt.figure(figsize=(10, 6))
                        if selected_chart_type == _("chart_type_hist"):
                            sns.histplot(df[selected_numeric_col].dropna(), kde=True)
                            plt.title(f"{selected_numeric_col} {_('histogram_title')}")
                            plt.xlabel(selected_numeric_col)
                            plt.ylabel(_("xlabel_freq"))
                        elif selected_chart_type == _("chart_type_line"):
                            # For line chart, assume index as x-axis or need a time column
                            plt.plot(df[selected_numeric_col])
                            plt.title(f"{selected_numeric_col} Trend")
                            plt.xlabel("Index")
                            plt.ylabel(selected_numeric_col)
                        elif selected_chart_type == _("chart_type_box"):
                            sns.boxplot(y=df[selected_numeric_col].dropna())
                            plt.title(f"{selected_numeric_col} Box Plot")
                            plt.ylabel(selected_numeric_col)
                        st.pyplot(plt)
                        plt.close() # Close plot to prevent display issues

                else:
                    st.info(_("warning_no_numeric_cols"))

            # Categorical Column Visualization
            st.markdown("---")
            with st.expander(_("expander_categorical_viz")):
                categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                if categorical_cols:
                    selected_cat_col = st.selectbox(
                        _("select_cat_col"),
                        categorical_cols,
                        key="cat_col_select"
                    )
                    if selected_cat_col:
                        plt.figure(figsize=(10, 6))
                        sns.countplot(y=df[selected_cat_col].dropna(), order=df[selected_cat_col].value_counts().index)
                        plt.title(f"{selected_cat_col} {_('bar_chart_title')}")
                        plt.xlabel(_("xlabel_freq"))
                        plt.ylabel(selected_cat_col)
                        st.pyplot(plt)
                        plt.close()
                else:
                    st.info(_("info_no_cat_cols"))

        except pd.errors.EmptyDataError:
            st.error(_("error_read_csv") + " " + _("info_check_csv_format") + " (Empty file)")
        except Exception as e:
            st.error(f"{_('error_read_csv')} {e}")
            st.info(_("info_check_csv_format"))
    else:
        st.info(_("info_upload_csv"))


# --- Gemini AI Chat Tab ---
with tab_gemini_ai:
    st.header(_("section_ai_assistant"))

    # Display Gemini model status in sidebar
    st.sidebar.subheader(_("sidebar_gemini_status"))
    st.sidebar.write(st.session_state.gemini_status)

    if st.session_state.gemini_model:
        st.write(_("ai_prompt"))

        # Initialize chat session if not already
        if "chat" not in st.session_state or st.session_state.chat is None:
            try:
                st.session_state.chat = st.session_state.gemini_model.start_chat(history=st.session_state.chat_history)
                st.session_state.chat_init_success = True
            except Exception as e:
                st.error(_("error_chat_init_failed"))
                st.error(f"Debug Info: {e}") # For debugging purposes
                st.session_state.chat_init_success = False
        else:
            st.session_state.chat.history = st.session_state.chat_history # Sync history

        # Display chat messages from history
        for message in st.session_state.chat_history:
            role = "user" if message.role == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(message.parts[0].text)

        # Chat input from user
        user_query = st.chat_input(_("chat_input_placeholder"))

        if user_query:
            if not st.session_state.chat_init_success:
                st.warning(_("error_chat_init_failed"))
            else:
                st.chat_message("user").markdown(user_query)
                st.session_state.chat_history.append(genai.types.contents.Content(parts=[genai.types.part_types.Part(text=user_query)], role='user'))

                with st.chat_message("assistant"):
                    with st.spinner(_("gemini_thinking")):
                        try:
                            response = st.session_state.chat.send_message(user_query, stream=True)
                            response_text = ""
                            for chunk in response:
                                response_text += chunk.text
                            st.markdown(_("gemini_response_title") + " " + response_text)
                            st.session_state.chat_history.append(genai.types.contents.Content(parts=[genai.types.part_types.Part(text=response_text)], role='model'))
                        except genai.types.generation_types.StopCandidateException as e:
                            st.error(_("error_gemini_no_response"))
                            st.info(_("info_content_policy"))
                            st.error(f"Debug Info: {e}")
                        except Exception as e:
                            st.error(_("error_gemini_communication"))
                            st.info(_("warning_api_or_network_policy") + " " + _("info_check_api_and_policy"))
                            st.error(f"Debug Info: {e}")

        # Clear chat history button
        if st.button(_("button_clear_chat")):
            st.session_state.chat_history = []
            st.session_state.chat = None # Reset chat session
            st.success(_("chat_clear_success"))
            st.rerun()

    else: # If gemini model is not loaded (e.g., no API key or init failed)
        st.info(_("warning_ai_unavailable"))
        st.info(_("info_check_api_key"))