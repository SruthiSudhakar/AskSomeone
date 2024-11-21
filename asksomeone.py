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
import gradio as gr

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
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


batch_size = 32
arxivdata = load_dataset(
    'jamescalam/llama-2-arxiv-papers-chunked',
    split='train',
    cache_dir='/local/zemel/datasets/'
)
appajidata = load_dataset(
    'ss6638/AppajiSpeeches',
    split='train',
    cache_dir='/local/zemel/datasets/'
)
swamisarvapriyanandaqadata = load_dataset(
    'ss6638/swamisarvapriyanandaqa',
    split='train',
    cache_dir='/local/zemel/datasets/'
)
data=appajidata
data = data.to_pandas()
print('datalen',len(data))

# swamis_index_name = 'llama-2-rag-swamisarvapriyanandaqa'
appaji_index_name = 'llama-2-rag-appaji-speeches'
docs = ["Jaya Guru Datta"]
embeddings = embed_model.embed_documents(docs)
print(f"We have {len(embeddings)} doc embeddings, each with "
      f"a dimensionality of {len(embeddings[0])}.")
# check if index already exists (it shouldn't if this is first time)
if appaji_index_name not in pc.list_indexes().names():
    print('creating index')
    # if does not exist, create index
    pc.create_index(
        appaji_index_name,
        dimension=len(embeddings[0]),
        metric='cosine',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(appaji_index_name).status['ready']:
        time.sleep(1)
index = pc.Index(appaji_index_name)
index.describe_index_stats()
for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['discourse_date']}-{str(i)}" for i, x in batch.iterrows()]
    texts = [x['discourse_chunk'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['discourse_chunk'], 
        'link': x['discourse_link'],
        'dates': x['discourse_date']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))
    print('UPSERTED THIS MANY:', len(ids))
index.describe_index_stats()


# connect to index
appaji_index = pc.Index(appaji_index_name)
# swamis_index = pc.Index(swamis_index_name)
# view index stats
appaji_index.describe_index_stats()
# swamis_index.describe_index_stats()

model_id = 'meta-llama/Llama-2-70b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    cache_dir='/local/zemel/datasets/'
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
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

llm = HuggingFacePipeline(pipeline=generate_text)

text_field = 'text'  # field in metadata that contains text content

vectorstore_appaji = LCVSPinecone(
    appaji_index, embed_model.embed_query, text_field
)
# vectorstore_swamis = LCVSPinecone(
#     swamis_index, embed_model.embed_query, text_field
# )
# appaji_rag_pipeline = RetrievalQA.from_chain_type(
#     llm=llm, chain_type='stuff',
#     retriever=vectorstore_appaji.as_retriever()
# )

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore_appaji.as_retriever()
)
def ask_appaji(question):
    answer = ""
    response = rag_pipeline(question)['result']
    answer += "Appaji: " + response.split('Helpful Answer: ')[-1]
    answer += "\n\n\nI used the following texts to answer the qustion: \n"
    sources = response.split('Question: ')[0].split('make up an answer.')[1].split('\n')
    idx = 0
    for source in sources:
        if len(source)> 5:
            answer += str(idx)+". " + source.strip('\n') + " \n"
            idx+=1
    return answer
demo = gr.Interface(
    fn=ask_appaji,
    inputs=["text"],
    outputs=["text"],
)
demo.launch(share=True)