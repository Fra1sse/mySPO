allow_tf32 = True
compare_func_cfg = dict(threshold=0.3, type='preference_score_compare')
dataloader_drop_last = False
dataloader_num_workers = 16
dataloader_pin_memory = True
dataloader_shuffle = True
dataset_cfg = dict(
    meta_json_path=
    'SPO-main/SPO-main/spo_training_and_inference/assets/prompts/100_training_prompts.json',
    pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    type='PromptDataset')
eval_interval = 1
logdir = 'work_dirs'
lora_rank = 4
num_checkpoint_limit = None
num_epochs = 10
num_validation_images = 2
preference_model_func_cfg = dict(
    ckpt_path='model_ckpts/sd-v1-5_step-aware_preference_model.bin',
    model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
    processor_pretrained_model_name_or_path=
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    type='step_aware_preference_model_func')
pretrained = dict(model='runwayml/stable-diffusion-v1-5')
resume_from = ''
run_name = 'spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10'
sample = dict(
    eta=1.0,
    guidance_scale=5.0,
    num_sample_each_step=4,
    num_steps=20,
    sample_batch_size=2)
save_interval = 1
seed = 42
train = dict(
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    adam_weight_decay=0.0001,
    beta=10.0,
    cfg=True,
    divert_start_step=4,
    eps=0.1,
    gradient_accumulation_steps=1,
    learning_rate=6e-05,
    max_grad_norm=1.0,
    train_batch_size=1,
    use_8bit_adam=True)
use_checkpointing = False
use_lora = True
use_xformers = False
validation_prompts = [
    'studio portrait of young female, curly hair, neutral expression, soft studio lighting, high-detail skin',
]
wandb_entity_name = None
wandb_project_name = 'spo'
