import streamlit as st
import pandas as pd
import time
import requests
import os
import random
from scholarly import scholarly

# --- 页面配置 ---
st.set_page_config(page_title="Google Scholar AI 文献助手 (OpenAI)", layout="wide")

# --- 初始化 Session State (用于存储抓取的数据) ---
if 'crawler_data' not in st.session_state:
    st.session_state.crawler_data = []

# --- 侧边栏：配置区 ---
st.sidebar.header("⚙️ 配置中心")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="从 https://platform.openai.com/ 申请")
openai_model = st.sidebar.selectbox("选择模型", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], index=1)

ai_guide = st.sidebar.text_area("AI 分类标准", value="""你是一名学术文献专家。请分析文献标题和摘要，将文献归类为以下之一：
1. 历史研究 (History)
2. 政策分析 (Policy)
3. 技术实现 (Technical)
4. 经济贸易 (Economy)
5. 其它 (Other)
请仅输出类别名称和代码，如 "1. 历史研究"。如果内容完全无关，请输出 "IGNORE"。""", height=200)

results_per_query = st.sidebar.slider("每个关键词抓取条数", 1, 100, 10)

# --- 主界面 ---
st.title("🎓 Google Scholar 文献抓取与 ChatGPT 智能分类")
st.markdown("""
使用说明：在左侧输入 OpenAI API Key，在下方输入关键词，点击“开始抓取”即可。
**注意：** 运行前请确保本地环境可以正常访问 OpenAI 和 Google Scholar（需科学上网）。
""")

search_input = st.text_area("请输入搜索关键词 (每行一个)", placeholder="\"Silk Road\" service\n一带一路 数字化", height=150)

# --- 核心函数 ---
def get_ai_category(title, snippet, key, guide, model):
    if not key: return "未配置Key"
    # OpenAI 标准接口地址
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": guide},
            {"role": "user", "content": f"标题: {title}\n摘要: {snippet}"}
        ],
        "temperature": 0.2
    }
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }
    
    try:
        # OpenAI 接口调用
        res = requests.post(url, json=payload, headers=headers, timeout=20)
        res_json = res.json()
        if "choices" in res_json:
            return res_json['choices'][0]['message']['content'].strip()
        else:
            return f"API错误: {res_json.get('error', {}).get('message', '未知错误')}"
    except Exception as e:
        return f"连接失败: {str(e)}"

# --- 运行逻辑 ---
if st.button("🚀 开始抓取任务"):
    if not openai_api_key:
        st.error("请先在左侧侧边栏配置 OpenAI API Key！")
    elif not search_input.strip():
        st.warning("请输入至少一个关键词。")
    else:
        queries = [q.strip() for q in search_input.split('\n') if q.strip()]
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty()
        
        total_steps = len(queries) * results_per_query
        current_step = 0
        
        for q_idx, query in enumerate(queries):
            status_text.info(f"正在检索关键词: {query} ...")
            try:
                search_query = scholarly.search_pubs(query)
            except Exception as e:
                st.error(f"访问 Google Scholar 失败: {e}。请检查代理设置。")
                break
                
            count = 0
            while count < results_per_query:
                try:
                    pub = next(search_query)
                    bib = pub.get('bib', {})
                    title = bib.get('title', 'N/A')
                    
                    # 避免当前会话内重复
                    if any(item['标题'] == title for item in st.session_state.crawler_data):
                        count += 1
                        continue
                        
                    snippet = bib.get('abstract', '无摘要')
                    # 调用 OpenAI 分类
                    category = get_ai_category(title, snippet, openai_api_key, ai_guide, openai_model)
                    
                    if "IGNORE" not in category.upper():
                        item = {
                            "分类": category,
                            "标题": title,
                            "作者": bib.get('author', 'N/A'),
                            "年份": bib.get('pub_year', 'N/A'),
                            "来源": bib.get('venue', 'N/A'),
                            "引用量": pub.get('num_citations', 0),
                            "链接": pub.get('pub_url', 'N/A'),
                            "摘要": snippet,
                            "搜索关键词": query
                        }
                        st.session_state.crawler_data.append(item)
                    
                    count += 1
                    current_step += 1
                    progress_bar.progress(min(current_step / total_steps, 1.0))
                    
                    # 实时刷新表格显示 (展示最后5条)
                    df_display = pd.DataFrame(st.session_state.crawler_data)
                    table_placeholder.dataframe(df_display.tail(5), use_container_width=True)
                    
                    # 模拟人类行为，减缓频率
                    time.sleep(random.uniform(3.0, 6.0))
                    
                except StopIteration:
                    break
                except Exception as e:
                    st.warning(f"单条目抓取异常: {e}")
                    time.sleep(5)
                    break
        
        status_text.success("任务完成！")

# --- 数据展示与下载 ---
if st.session_state.crawler_data:
    st.divider()
    st.subheader(f"📊 抓取结果预览 (总计: {len(st.session_state.crawler_data)} 条)")
    final_df = pd.DataFrame(st.session_state.crawler_data)
    st.dataframe(final_df, use_container_width=True)
    
    # 导出 CSV
    csv_data = final_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="📥 下载完整数据 (CSV)",
        data=csv_data,
        file_name="google_scholar_openai_results.csv",
        mime="text/csv",
    )
    
    if st.button("🗑️ 清空所有数据"):
        st.session_state.crawler_data = []
        st.rerun()