import streamlit as st
import os
from dotenv import load_dotenv

# LlamaIndex関連
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# .envを読み込む
load_dotenv()

# ==============================
# ① インデックスをキャッシュ
# ==============================

@st.cache_resource
def load_index():
    # LLM
    llm = OpenAI(model="gpt-4o-mini")

    # Embedding
    embedding_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )

    # データ読み込み
    documents = SimpleDirectoryReader(
        input_dir="data",
        required_exts=[".txt"],
    ).load_data()

    # チャンク分割
    parser = SimpleNodeParser.from_defaults(
        chunk_size=300,
        chunk_overlap=50
    )

    nodes = parser.get_nodes_from_documents(documents)

    # インデックス作成
    index = VectorStoreIndex(
        nodes,
        embedding_model=embedding_model
    )

    return index, llm, len(documents), len(nodes)

# ==============================
# ② 初期読み込み
# ==============================

index, llm, doc_count, node_count = load_index()

# ==============================
# ③ UI
# ==============================

st.title("🧠 健康RAGアプリ（運動・食事・睡眠）")

st.write("質問を入力すると、健康に関する情報をもとに回答します。")



# 入力
question = st.text_input("質問を入力してください 🔍")

# ==============================
# ④ 質問処理
# ==============================

if question:

    # クエリエンジン
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        llm=llm
    )

    # ローディング表示
    with st.spinner("検索中..."):
        response = query_engine.query(question)

    st.divider()
    
    st.write(f"📄 ドキュメント数: {doc_count} / 🔹 チャンク数: {node_count}")
    
    # ==============================
    # ⑤ 回答判定
    # ==============================

    threshold = 0.85

    valid_nodes = [
        node for node in response.source_nodes
        if node.score >= threshold
    ]
    
    st.subheader("🎓 回答")

    if len(valid_nodes) == 0:
        st.error("関連性の高い情報が見つからなかったため、回答できませんでした。")

    elif len(valid_nodes) < 2:
        st.warning("十分な根拠となる情報が揃わなかったため、回答を控えました。")

    else:
        st.success(response.response)

    # ==============================
    # ⑥ 参考情報
    # ==============================

    st.subheader("💡 参考情報")

    for i, node in enumerate(response.source_nodes):

        with st.expander(f"参考 {i+1}"):

            # スコア
            st.write(f"類似度スコア: {round(node.score, 3)}")

            # 出典
            source = node.node.metadata.get("file_name", "不明")
            st.write(f"出典ファイル: {source}")

            # 本文（改行削除）
            text = node.node.text.replace("\n", " ")
            st.write("関連文章:")
            st.write(text[:200])