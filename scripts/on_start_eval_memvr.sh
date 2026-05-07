cd "$DATA_DIRECTORY"
git clone -b memvr https://github.com/jden4524/attn_ft.git
cd attn_ft/eval
git clone https://github.com/hengzhan/VLMEvalKit.git
uv venv eval
source eval/bin/activate
uv pip install -r ./VLMEvalKit/requirements.txt
uv pip install -e ./VLMEvalKit