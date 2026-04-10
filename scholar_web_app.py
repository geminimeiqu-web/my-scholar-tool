import streamlit as st
import pandas as pd
import time
import requests
import random
import json
from scholarly import scholarly

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Google Scholar 全能 AI 助手",
    page_icon="🎓",
    layout="wide"
)

# --- 样式美化与颜色修复 ---
st.markdown("""
    <style>
    /* 页面主背景 */
    .stApp { background-color: #f8f9fa; }
    
    /* 强制修复输入框、文本域的文字颜色和背景色 */
    .stTextArea textarea, .stTextInput input {
        color: #1f2937 !important; /* 深灰色文字 */
        background-color: #ffffff !important; /* 纯白背景 */
        border: 1px solid #d1d5db !important;
    }
    
    /* 修复标签（标题）文字颜色 */
    .stMarkdown p, label, .stSelectbox label {
        color: #374151 !important;
        font-weight: 500;
    }

    /* 侧边栏背景与文字 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007BFF;
        color: white !important;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 初始化数据存储 ---
if 'crawler_results' not in st.session_state:
    st.session_state.crawler_results = []

# --- 侧边栏：配置中心 ---
with st.sidebar:
    st.title("🛡️ 接口配置")
    
    # 1. 选择 AI 服务商
    provider = st.selectbox(
        "选择 AI 服务商",
        ["ChatGPT (OpenAI)", "DeepSeek", "Claude (Anthropic)", "Gemini (Google)"],
        help="请选择您想要使用的模型品牌"
    )
    
    # 2. 动态模型参数配置
    if provider == "ChatGPT (OpenAI)":
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        model_name = st.selectbox("选择模型", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    elif provider == "DeepSeek":
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")
        model_name = st.selectbox("选择模型", ["deepseek-chat", "deepseek-reasoner"])
    elif provider == "Claude (Anthropic)":
        api_key = st.text_input("Claude API Key", type="password", placeholder="sk-ant-...")
        model_name = st.selectbox("选择模型", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"])
    elif provider == "Gemini (Google)":
        api_key = st.text_input("Gemini API Key", type="password", placeholder="直接输入 Key")
        model_name = st.selectbox("选择模型", ["gemini-1.5-flash", "gemini-1.5-pro"])
        
    st.divider()
    
    # 3. 抓取控制
    results_limit = st.slider("每个词抓取上限", 1, 50, 5)
    
    # 4. 提示词定义 (AI指令)
    system_prompt = st.text_area(
        "AI 角色指令 (Prompt)",
        value="你是一名学术专家。请根据标题和摘要分析文献，将其归类为：1.基础理论 2.应用实践 3.政策研究 4.技术开发 5.其它。仅输出类别名称和简短理由（20字内）。若不相关输出 IGNORE。",
        height=180
    )

# --- 核心 AI 调用逻辑 ---
def call_ai_unified(title, abstract, key, provider, model, prompt):
    if not key: return "未配置 Key"
    
    user_content = f"标题: {title}\n摘要: {abstract}"
    
    try:
        # 1. OpenAI 与 DeepSeek
        if provider in ["ChatGPT (OpenAI)", "DeepSeek"]:
            url = "https://api.openai.com/v1/chat/completions" if provider == "ChatGPT (OpenAI)" else "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_content}],
                "temperature": 0.3
            }
            res = requests.post(url, json=payload, headers=headers, timeout=25)
            return res.json()['choices'][0]['message']['content'].strip()

        # 2. Claude
        elif provider == "Claude (Anthropic)":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": model,
                "system": prompt,
                "messages": [{"role": "user", "content": user_content}],
                "max_tokens": 500,
                "temperature": 0.3
            }
            res = requests.post(url, json=payload, headers=headers, timeout=25)
            return res.json()['content'][0]['text'].strip()

        # 3. Gemini
        elif provider == "Gemini (Google)":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": f"{prompt}\n\n输入内容：{user_content}"}]}]
            }
            res = requests.post(url, json=payload, headers=headers, timeout=25)
            return res.json()['candidates'][0]['content']['parts'][0]['text'].strip()

    except Exception as e:
        return f"接口连接异常: {str(e)}"

# --- 主界面布局 ---
st.title("🎓 学术文献 AI 自动化处理中心")
st.caption("集成了主流四大模型，实现 Google Scholar 文献的实时检索、分类与摘要分析")

# 关键词输入
query_input = st.text_area(
    "输入检索关键词 (每行一个词)",
    placeholder="例：\n\"Artificial Intelligence\" Ethics\nDigital Economy Silk Road\n碳中和 路径研究",
    height=150
)

col1, col2 = st.columns([3, 1])
with col1:
    btn_start = st.button("🚀 启动自动化流水线")
with col2:
    if st.button("🗑️ 清空所有缓存"):
        st.session_state.crawler_results = []
        st.rerun()

# --- 自动化流程逻辑 ---
if btn_start:
    if not api_key:
        st.error(f"❌ 请输入 {provider} 的 API Key 才能继续")
    elif not query_input.strip():
        st.warning("⚠️ 请输入搜索关键词")
    else:
        queries = [q.strip() for q in query_input.split('\n') if q.strip()]
        progress_bar = st.progress(0)
        status = st.empty()
        preview = st.empty()
        
        for q_idx, q in enumerate(queries):
            status.info(f"正在处理关键词: **{q}** ...")
            try:
                search_iter = scholarly.search_pubs(q)
            except Exception as e:
                st.error(f"无法访问 Google Scholar: {e}")
                break
            
            count = 0
            while count < results_limit:
                try:
                    pub = next(search_iter)
                    bib = pub.get('bib', {})
                    title = bib.get('title', 'N/A')
                    
                    if any(r['标题'] == title for r in st.session_state.crawler_results):
                        count += 1
                        continue
                    
                    abstract = bib.get('abstract', '无摘要内容')
                    ai_result = call_ai_unified(title, abstract, api_key, provider, model_name, system_prompt)
                    
                    if "IGNORE" not in ai_result.upper():
                        record = {
                            "AI 分类结果": ai_result,
                            "标题": title,
                            "年份": bib.get('pub_year', 'N/A'),
                            "作者": bib.get('author', 'N/A'),
                            "被引": pub.get('num_citations', 0),
                            "来源": bib.get('venue', 'N/A'),
                            "链接": pub.get('pub_url', 'N/A'),
                            "摘要": abstract,
                            "AI 服务商": provider
                        }
                        st.session_state.crawler_results.append(record)
                        df_preview = pd.DataFrame(st.session_state.crawler_results)
                        preview.dataframe(df_preview.tail(3), use_container_width=True)
                    
                    count += 1
                    time.sleep(random.uniform(2.5, 4.5))
                    
                except StopIteration:
                    break
                except Exception as e:
                    st.warning(f"跳过单条故障条目: {e}")
                    break
            
            progress_bar.progress((q_idx + 1) / len(queries))
            
        status.success("🎉 任务圆满完成！")

# --- 最终报表与导出 ---
if st.session_state.crawler_results:
    st.divider()
    df_final = pd.DataFrame(st.session_state.crawler_results)
    st.subheader(f"📈 处理结果报表 (共计 {len(df_final)} 条数据)")
    st.dataframe(df_final, use_container_width=True)
    
    csv_out = df_final.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="📥 下载文献分类分析报告 (CSV/Excel)",
        data=csv_out,
        file_name="scholarly_ai_report.csv",
        mime="text/csv"
    )
