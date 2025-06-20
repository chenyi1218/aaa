import google.generativeai as genai
import os

# 從 secrets.toml 模擬讀取金鑰，或直接在這裡貼上金鑰進行測試
# 更好的方法是讀取 secrets.toml
# 這裡為了快速測試，假設你直接從環境變量讀取或硬編碼（不推薦實際應用）
# 如果你確定金鑰在 secrets.toml，可以在這裡先手動複製貼上測試
# 從 secrets.toml 中讀取，需要安裝toml庫：pip install toml
import toml

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
    api_key = secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    print("API 金鑰已配置。")
except FileNotFoundError:
    print("錯誤：找不到 .streamlit/secrets.toml 檔案。")
    exit()
except KeyError:
    print("錯誤：secrets.toml 中找不到 GEMINI_API_KEY。")
    exit()
except Exception as e:
    print(f"配置金鑰時發生錯誤: {e}")
    exit()

target_model_name = "models/gemini-1.5-flash" # 使用您希望測試的模型

try:
    print(f"嘗試列出模型以確認連線...")
    all_models = genai.list_models()
    model_found = False
    for m in all_models:
        if m.name == target_model_name:
            model_found = True
            print(f"發現目標模型: {m.name}")
            print(f"支持的生成方法: {m.supported_generation_methods}")
            if "generateContent" in m.supported_generation_methods:
                print(f"模型 '{target_model_name}' 支持 generateContent。")
                # 嘗試創建模型實例
                model = genai.GenerativeModel(target_model_name)
                print(f"模型 '{target_model_name}' 成功載入！")
            else:
                print(f"錯誤：模型 '{target_model_name}' 不支持 generateContent。")
    if not model_found:
        print(f"錯誤：在可用模型列表中找不到 '{target_model_name}'。")

except Exception as e:
    print(f"連接 Gemini API 或載入模型時發生錯誤: {e}")
    print("請檢查：")
    print("  1. 您的 API 金鑰是否有效且有權限。")
    print("  2. 您的網路連線是否正常。")
    print("  3. 模型名稱是否正確，且該模型在您所在的地區是否可用。")