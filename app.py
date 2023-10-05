import streamlit as st
import pandas as pd
from utils import extract_from_url, get_model, calculate_memory
import plotly.express as px
import numpy as np
import gc


st.set_page_config(page_title='Can you run it? LLM version', layout="wide", initial_sidebar_state="expanded")

st.title("Can you run it? LLM version")

percentage_width_main = 80
st.markdown(
        f"""<style>
        .appview-container .main .block-container{{
        max-width: {percentage_width_main}%;}}
        </style>
        """,
        unsafe_allow_html=True,
    )
@st.cache_resource
def get_gpu_specs():
    return pd.read_csv("data/gpu_specs.csv")

@st.cache_resource
def get_mistralai_table():
    model = get_model("mistralai/Mistral-7B-v0.1", library="transformers", access_token="")
    return calculate_memory(model, ["float32", "float16/bfloat16", "int8", "int4"])

def show_gpu_info(info, trainable_params=0):
    for var in ['Inference', 'Full Training Adam', 'LoRa Fine-tuning']:
        _info = info.loc[var]
        if _info['Number of GPUs'] >= 3:
            func = st.error
            icon = "⛔"
        elif _info['Number of GPUs'] == 2:
            func = st.warning
            icon = "⚠️"
        else:
            func = st.success
            icon = "✅"
        
        msg = f"You require **{_info['Number of GPUs']}** GPUs for **{var}**"
        if var == 'LoRa Fine-tuning':
            msg += f" ({trainable_params}%)"
        func(msg, icon=icon)


def get_name(index):
    row = gpu_specs.iloc[index]
    return f"{row['Product Name']} ({row['RAM (GB)']} GB, {row['Year']})"

def custom_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)
gpu_specs = get_gpu_specs()

_, col, _ = st.columns([1,3,1])
with col.expander("Information", expanded=True):
    st.markdown("""- GPU information comes from [TechPowerUp GPU Specs](https://www.techpowerup.com/gpu-specs/)
- Mainly based on [Model Memory Calculator by hf-accelerate](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
    using `transformers` library
- Inference is calculated following [EleutherAI Transformer Math 101](https://blog.eleuther.ai/transformer-math/),
    where is estimated as """)
    
    st.latex(r"""\text{Memory}_\text{Inference} \approx \text{Model Size} \times 1.2""")
    st.markdown("""- For LoRa Fine-tuning, I'm asuming a **16-bit** dtype of trainable parameters. The formula (in terms of GB) is""")
    st.latex(r"\text{Memory}_\text{LoRa} \approx \text{Model Size} + \left(\text{ \# trainable Params}_\text{Billions}\times\frac{16}{8} \times 4\right) \times 1.2")
    st.markdown("- You can understand `int4` as models in `GPTQ-4bit`, `AWQ-4bit` or `Q4_0 GGUF/GGML` formats")

access_token = st.sidebar.text_input("Access token")
model_name = st.sidebar.text_input("Model name", value="mistralai/Mistral-7B-v0.1")
if not model_name:
    st.info("Please enter a model name")
    st.stop()

model_name = extract_from_url(model_name)
if model_name not in st.session_state:
    if 'actual_model' in st.session_state:
        del st.session_state[st.session_state['actual_model']]
        del st.session_state['actual_model']
        gc.collect()
    if model_name == "mistralai/Mistral-7B-v0.1": # cache Mistral
        st.session_state[model_name] = get_mistralai_table()
    else:
        model = get_model(model_name, library="transformers", access_token=access_token)
        st.session_state[model_name] = calculate_memory(model, ["float32", "float16/bfloat16", "int8", "int4"])
        del model
        gc.collect()
    st.session_state['actual_model'] = model_name


gpu_vendor = st.sidebar.selectbox("GPU Vendor", ["NVIDIA", "AMD", "Intel"])
# year = st.sidebar.selectbox("Filter by Release Year", list(range(2014, 2024))[::-1], index=None)
gpu_info = gpu_specs[gpu_specs['Vendor'] == gpu_vendor].sort_values('Product Name')
# if year:
#     gpu_info = gpu_info[gpu_info['Year'] == year]

min_ram = gpu_info['RAM (GB)'].min()
max_ram = gpu_info['RAM (GB)'].max()
ram = st.sidebar.slider("Filter by RAM (GB)", min_ram, max_ram, (10.0, 40.0), step=0.5)
gpu_info = gpu_info[gpu_info["RAM (GB)"].between(ram[0], ram[1])]
if len(gpu_info) == 0:
    st.sidebar.error(f"**{gpu_vendor}** has no GPU in that RAM range")
    st.stop()
gpu = st.sidebar.selectbox("GPU", gpu_info['Product Name'].index.tolist(), format_func=lambda x : gpu_specs.iloc[x]['Product Name'])
gpu_spec = gpu_specs.iloc[gpu]
gpu_spec.name = 'INFO'

lora_pct = st.sidebar.slider("LoRa % trainable parameters", 0.1, 100.0, 2.0, step=0.1)

st.sidebar.dataframe(gpu_spec.T.astype(str))

memory_table = pd.DataFrame(st.session_state[model_name]).set_index('dtype')
memory_table['LoRA Fine-Tuning (GB)'] = (memory_table["Total Size (GB)"] + 
                                          (memory_table["Parameters (Billion)"]* lora_pct/100 * (16/8)*4)) * 1.2
    
_memory_table = memory_table.copy()
memory_table = memory_table.round(2).T
_memory_table /= gpu_spec['RAM (GB)']
_memory_table = _memory_table.apply(np.ceil).astype(int).drop(columns=['Parameters (Billion)', 'Total Size (GB)'])
_memory_table.columns = ['Inference', 'Full Training Adam', 'LoRa Fine-tuning']
_memory_table = _memory_table.stack().reset_index()
_memory_table.columns = ['dtype', 'Variable', 'Number of GPUs']
col1, col2 = st.columns([1,1.3])
with col1:
    st.write(f"####  [{model_name}](https://huggingface.co/{model_name}) ({custom_ceil(memory_table.iloc[3,0],1):.1f}B)")

    dtypes = memory_table.columns.tolist()[::-1]
    tabs = st.tabs(dtypes)
    for dtype, tab in zip(dtypes, tabs):
        with tab:
            info = _memory_table[_memory_table['dtype'] == dtype].set_index('Variable')
            show_gpu_info(info, lora_pct)
    st.write(memory_table.iloc[[0, 1, 2, 4]])
with col2:
    num_colors= 4
    colors = [px.colors.sequential.RdBu[int(i*(len(px.colors.sequential.RdBu)-1)/(num_colors-1))] for i in range(num_colors)]
    fig = px.bar(_memory_table, x='Variable', y='Number of GPUs', color='dtype', barmode='group', color_discrete_sequence=colors)
    fig.update_layout(title=dict(text=f"Number of GPUs required for<br> {get_name(gpu)}", font=dict(size=25))
                    , xaxis_tickfont_size=14, yaxis_tickfont_size=16, yaxis_dtick='1')
    st.plotly_chart(fig, use_container_width=True)
