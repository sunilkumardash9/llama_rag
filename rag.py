from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
import tiktoken
llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()
text_splitter = TokenTextSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"],
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
node_parser = SimpleNodeParser.from_defaults(
  text_splitter=text_splitter
)
prompt_helper = PromptHelper(
  context_window=4096, 
  num_output=256, 
  chunk_overlap_ratio=0.1, 
  chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)

documents = SimpleDirectoryReader(input_dir='data').load_data()
index = VectorStoreIndex.from_documents(
    documents, 
    service_context = service_context
    )
index.storage_context.persist()

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What is HNSW?")
print(response)
