cd "$DATA_DIRECTORY"
git clone -b memvr https://github.com/jden4524/attn_ft.git
cd attn_ft/eval
git clone -b "${VLMEVALKIT_BRANCH:-main}" "${VLMEVALKIT_REPO:?Set VLMEVALKIT_REPO to your VLMEvalKit fork URL}" VLMEvalKit
uv venv eval
source eval/bin/activate
uv pip install -r ./VLMEvalKit/requirements.txt
uv pip install -e ./VLMEvalKit