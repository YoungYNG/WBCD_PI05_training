data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_eef_data_to_lerobot.py --raw-dir "$data_dir" --repo-id "$repo_id"
