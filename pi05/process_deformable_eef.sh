raw_dir=${1:-"data/Deformable Manipulation"}
output_dir=${2:-"processed_data_eef/deformable_manipulation_eef"}
max_episodes=${3:-}
resume_flag=${4:-}

if [ -z "$max_episodes" ]; then
    python scripts/process_deformable_eef_data.py --raw-dir "$raw_dir" --output-dir "$output_dir" $resume_flag
else
    python scripts/process_deformable_eef_data.py --raw-dir "$raw_dir" --output-dir "$output_dir" --max-episodes "$max_episodes" $resume_flag
fi
