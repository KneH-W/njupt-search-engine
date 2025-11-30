import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- 1. 页面配置 (必须是第一行) ---
st.set_page_config(
    page_title="南邮新闻搜",
    page_icon="🎓",
    layout="centered"
)

# --- 2. 核心逻辑 (带缓存优化) ---
# @st.cache_data 是 Streamlit 的神器
# 它的作用是：只有第一次运行会加载数据和训练模型，后续刷新页面直接用缓存
# 否则用户每搜一次都要重新训练模型，速度会很慢
@st.cache_data
def load_data_and_model():
    # A. 读取数据
    try:
        df = pd.read_csv("njupt_news_cut.csv", keep_default_na=False)
    except FileNotFoundError:
        return None, None, None
    
    corpus = df['cut_content'].values
    
    # B. 训练模型
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return df, vectorizer, tfidf_matrix

# 加载资源
df, vectorizer, tfidf_matrix = load_data_and_model()

# --- 3. 侧边栏 (项目介绍 - 你的简历亮点) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/zh/4/44/Logo_of_NJUPT.svg", width=200)
    st.markdown("## 关于本项目")
    st.write("这是一个基于 **TF-IDF** 算法的垂直搜索引擎，专为检索南邮校内新闻设计。")
    
    st.markdown("### 🛠️ 技术栈")
    st.markdown("- **爬虫**: Requests + BeautifulSoup")
    st.markdown("- **数据清洗**: Pandas + Jieba")
    st.markdown("- **核心算法**: Scikit-learn (TF-IDF + Cosine Similarity)")
    st.markdown("- **界面**: Streamlit")
    
    st.markdown("---")
    st.write("Designed by 南邮大三学生")

    

# --- 4. 主界面 (UI) ---
st.title("🎓 南邮校内新闻搜索引擎")
st.markdown("输入关键词，瞬间找回丢失的校园记忆...")

# 检查数据是否加载成功
if df is None:
    st.error("❌ 错误：找不到 njupt_news_cut.csv！请先运行 Level 2 的清洗脚本。")
    st.stop()

# 搜索框
query = st.text_input("请输入关键词 (如：奖学金、考研、食堂)", placeholder="Try searching '计算机'...")
search_btn = st.button("🔍 立即搜索")

# --- 5. 搜索响应逻辑 ---
if search_btn and query:
    start_ts = time.time()
    
    # A. 处理查询词
    query_cut = " ".join(jieba.lcut(query))
    
    # B. 向量化 & 计算相似度
    query_vec = vectorizer.transform([query_cut])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # C. 排序取前 10
    sorted_indices = sim_scores.argsort()[::-1][:10]
    
    # D. 展示结果
    st.markdown("### 📊 搜索结果")
    
    found_count = 0
    for idx in sorted_indices:
        score = sim_scores[idx]
        if score < 0.05: continue # 过滤低相关性
        
        found_count += 1
        row = df.iloc[idx]
        
        # 使用 Streamlit 的 container 美化展示
        with st.container():
            # 标题带链接
            st.markdown(f"#### [{row['title']}]({row['link']})")
            
            # 显示匹配度进度条
            st.progress(float(score), text=f"相关度: {score:.2f}")
            
            # 摘要
            content_preview = str(row['content'])[:80] + "..."
            st.caption(content_preview)
            
            st.divider() # 分割线
            
    if found_count == 0:
        st.warning(f"没有找到关于 '{query}' 的新闻，换个词试试？")
    else:
        cost = time.time() - start_ts
        st.success(f"共找到 {found_count} 条相关结果，耗时 {cost:.4f} 秒")
