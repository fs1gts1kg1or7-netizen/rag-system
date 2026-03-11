import os
from dotenv import load_dotenv

# LlamaIndex関連
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser

# .envを読み込む
load_dotenv()

# ==============================
# ① LLM設定
# ==============================

llm = OpenAI(model="gpt-4o-mini")

# ==============================
# ② データ読み込み
# ==============================

documents = SimpleDirectoryReader(
    input_dir = "data",
    required_exts = [".txt"],  # .txtファイルのみ
).load_data()

print("読み込んだドキュメント数:", len(documents))

# ==============================
# ③ チャンク分割
# ==============================

parser = SimpleNodeParser.from_defaults(
    chunk_size=300,
    chunk_overlap=50
)

nodes = parser.get_nodes_from_documents(documents)

print("作成されたチャンク数:", len(nodes))

# ==============================
# ④ ベクトルインデックス作成
# ==============================

index = VectorStoreIndex(nodes)

# ==============================
# ⑤ クエリエンジン作成
# ==============================

query_engine = index.as_query_engine(
    similarity_top_k=5,
    llm=llm
)

# ==============================
# ⑥ 質問
# ==============================

question = "糖尿病のリスクについて教えてください"

response = query_engine.query(question)

# ==============================
# ⑦ 回答表示
# ==============================

print("\n===== 質問 =====")
print(question)

print("\n===== 回答 =====")
print(response)

# ==============================
# ⑧ 根拠となった文章を表示
# ==============================

print("\n===== 参考情報（検索されたチャンク） =====")

for i, node in enumerate(response.source_nodes):
    
    print(f"\n--- 参考 {i+1} ---")
    
    # 検索スコア
    print("類似度スコア:", node.score)
    
    # 元のファイル名
    source = node.node.metadata.get("file_name", "不明")
    
    print("出典ファイル:", source)
    print("関連文章:")
    
    print(node.node.text[:300])