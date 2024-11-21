from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from datasets import load_dataset
from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone as LCVSPinecone
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os, time, pdb
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
import gradio
"""
SETUP
"""
print('SETUP STARTED')
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:0' if cuda.is_available() else 'cpu'
print('CUDA USED???:', device)
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = '4b46f5d6-ba2e-4fc7-950d-c64897e4ed02'

# configure client
pc = Pinecone(api_key=api_key)
cloud = 'aws' # os.environ.get('PINECONE_CLOUD') or
region = 'us-east-1' #os.environ.get('PINECONE_REGION') or
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'llama-2-rag-swamisarvapriyanandaqa'
docs = ["Jaya Guru Datta"]
embeddings = embed_model.embed_documents(docs)
print(f"We have {len(embeddings)} doc embeddings, each with a dimensionality of {len(embeddings[0])}.")
# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    print('creating index')
    # if does not exist, create index
    pc.create_index(index_name,dimension=len(embeddings[0]),metric='cosine',spec=spec)
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()


"""
LOAD DATA
"""
print('LOADING DATA')
# arxivdata = load_dataset(
#     'jamescalam/llama-2-arxiv-papers-chunked',
#     split='train',
#     cache_dir='/local/zemel/datasets/'
# )
# appajidata = load_dataset(
#     'ss6638/AppajiSpeeches',
#     split='train',
#     cache_dir='/local/zemel/datasets/'
# )
data = load_dataset(
    'ss6638/swamisarvapriyanandaqa',
    split='train',
    cache_dir='/local/zemel/datasets/'
)

data = data.to_pandas()
batch_size = 16
print('datalen',len(data))
for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['discourse_date']}-{str(i)}" for i, x in batch.iterrows()]
    texts = [x['discourse_chunk'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['discourse_chunk'],
         'dates': x['discourse_date']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))
    print('UPSERTED THIS MANY:', len(ids))
index.describe_index_stats()

"""
LOAD MODEL
"""
print('LOADING MODEL')
model_id = 'meta-llama/Llama-2-70b-chat-hf'
hf_auth = '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'
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    cache_dir='/local/zemel/datasets/'
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    trust_remote_code=True,
    config=model_config,
    # quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
    cache_dir='/local/zemel/datasets/'
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# res = generate_text("Explain to me the difference between nuclear fission and fusion.")
# print('generated text': res[0]["generated_text"])


llm = HuggingFacePipeline(pipeline=generate_text)


"""
INITIALIZE A RETRIEVALQA CHAIN
"""
print('INTIALIZING RQA CHAIN')
text_field = 'text'  # field in metadata that contains text content
vectorstore = LCVSPinecone(
    index, embed_model.embed_query, text_field
)
template = """Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. Do not make up information, if you do not know an answer, just say 'I don't know': ------ <ctx> {context} </ctx> ------ <hs> {history} </hs> ------ {question} Answer: """

prompt = PromptTemplate(input_variables=["history", "context", "question"],template=template,)
prompt.input_variables=["history", "context", "question"]
memory = ConversationBufferMemory(memory_key="history",input_key="question")
chain_type_kwargs={"verbose": True,"prompt": prompt,"memory": memory}
qa = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=vectorstore.as_retriever(),verbose=True,chain_type_kwargs=chain_type_kwargs)

def reset():
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=vectorstore.as_retriever(),verbose=True,chain_type_kwargs=chain_type_kwargs)
def ask(question):
    print()
    print()
    print(qa.run(question))

    
print('hi! Ask me any questions!')
pdb.set_trace()

ask('What shlokas talk about karma in the Gita?')
reset()