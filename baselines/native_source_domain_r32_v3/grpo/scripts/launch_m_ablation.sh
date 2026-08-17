cd /data/GRPO/scripts
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 nohup /data/venvs/llamafactory-01398eb-liger081/bin/python run_nothink_m_ablation.py --m 4 --device cuda:0 --out /data/GRPO/logs/nothink_m4.json > /data/GRPO/logs/nothink_m4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 nohup /data/venvs/llamafactory-01398eb-liger081/bin/python run_nothink_m_ablation.py --m 8 --device cuda:0 --out /data/GRPO/logs/nothink_m8.json > /data/GRPO/logs/nothink_m8.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 nohup /data/venvs/llamafactory-01398eb-liger081/bin/python run_nothink_m_ablation.py --m 16 --device cuda:0 --out /data/GRPO/logs/nothink_m16.json > /data/GRPO/logs/nothink_m16.log 2>&1 &
echo ALL_STARTED