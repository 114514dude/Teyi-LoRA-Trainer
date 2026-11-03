const file = (key, zh, en, required = true) => ({ key, zh, en, kind: 'file', required })
const folder = (key, zh, en, required = true) => ({ key, zh, en, kind: 'folder', required })

export const MODEL_PROFILES = {
  'Anima': { backendType: 'anima', supportsBlockSwap: true, supportsTimestepSampling: true, lrMode: 'anima', fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('llm_path', 'Qwen3 0.6B 权重', 'Qwen3 0.6B weights')] },
  'Flux.1': { backendType: 'flux', supportsBlockSwap: true, supportsTimestepSampling: true, defaults: { guidance: 1 }, fields: [folder('diffusers_path', 'Diffusers 模型目录', 'Diffusers model folder'), file('transformer_path', 'Transformer 覆盖权重（可选）', 'Transformer override (optional)', false)] },
  'Flux.2': { backendType: 'flux2', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'flux2', comfy: true, textEncoderArray: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
  'Stable Diffusion XL': { backendType: 'sdxl', supportsBlockSwap: false, supportsTimestepSampling: false, lrMode: 'sdxl', fields: [file('checkpoint_path', 'SDXL 模型权重', 'SDXL checkpoint')] },
  'Stable Diffusion 3 / 3.5': { backendType: 'sd3', fields: [folder('diffusers_path', 'Diffusers 模型目录', 'Diffusers model folder')] },
  'Qwen Image': { backendType: 'qwen_image', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('diffusers_path', 'Qwen Image 模型目录', 'Qwen Image model folder')] },
  'Qwen Image 2511': { backendType: 'qwen2511', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('diffusers_path', 'Qwen 2511 模型目录', 'Qwen 2511 model folder')] },
  'Wan 2.1 / 2.2': { backendType: 'wan', supportsBlockSwap: true, supportsTimestepSampling: false, fields: [folder('ckpt_path', 'Wan 官方模型目录', 'Wan checkpoint folder'), file('transformer_path', 'Transformer 覆盖权重（可选）', 'Transformer override (optional)', false)] },
  'HunyuanVideo': { backendType: 'hunyuan-video', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('ckpt_path', 'HunyuanVideo 官方模型目录', 'HunyuanVideo checkpoint folder')] },
  'HunyuanVideo 1.5': { backendType: 'hunyuan_video_15', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'hunyuan_video_15', comfy: true, textEncoderArray: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
  'HunyuanImage 2.1': { backendType: 'hunyuan_image', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('text_encoder_path', '文本编码器权重', 'Text encoder weights'), file('byt5_path', 'ByT5 权重', 'ByT5 weights')] },
  'LTX Video': { backendType: 'ltx-video', fields: [folder('diffusers_path', 'LTX Diffusers 目录', 'LTX Diffusers folder'), file('single_file_path', 'LTX 单文件权重', 'LTX single-file weights')] },
  'LTX-2': { backendType: 'ltx2', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'ltxv', comfy: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
  'Lumina Image 2': { backendType: 'lumina_2', defaults: { lumina_shift: true }, fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('llm_path', 'Gemma 文本编码器权重', 'Gemma text encoder weights')] },
  'Chroma': { backendType: 'chroma', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('diffusers_path', 'Flux Diffusers 基础目录', 'Flux Diffusers base folder'), file('transformer_path', 'Chroma Transformer 权重', 'Chroma transformer weights')] },
  'Cosmos Predict2': { backendType: 'cosmos_predict2', supportsBlockSwap: true, supportsTimestepSampling: true, lrMode: 'anima', fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('t5_path', 'T5 编码器权重', 'T5 encoder weights')] },
  'Z-Image': { backendType: 'z_image', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('checkpoint_path', 'Z-Image 模型目录', 'Z-Image checkpoint folder')] },
  'AuraFlow': { backendType: 'auraflow', supportsBlockSwap: true, supportsTimestepSampling: true, defaults: { max_sequence_length: 256 }, fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('text_encoder_path', '文本编码器权重', 'Text encoder weights')] },
  'HiDream': { backendType: 'hidream', supportsBlockSwap: true, supportsTimestepSampling: true, fields: [folder('diffusers_path', 'HiDream Diffusers 目录', 'HiDream Diffusers folder'), folder('llama3_path', 'Llama 3 模型目录', 'Llama 3 model folder')] },
  'OmniGen2': { backendType: 'omnigen2', fields: [folder('diffusers_path', 'OmniGen2 模型目录', 'OmniGen2 model folder')] },
  'Cosmos': { backendType: 'cosmos', supportsBlockSwap: false, supportsTimestepSampling: false, fields: [file('transformer_path', 'Transformer 权重', 'Transformer weights'), file('vae_path', 'VAE 权重', 'VAE weights'), file('text_encoder_path', '文本编码器权重', 'Text encoder weights')] },
  'ERNIE Image': { backendType: 'ernie_image', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'stable_diffusion', comfy: true, textEncoderArray: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
  'Ideogram 4': { backendType: 'ideogram4', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'ideogram4', comfy: true, textEncoderArray: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
  'Krea 2': { backendType: 'krea2', supportsBlockSwap: true, supportsTimestepSampling: true, textEncoderType: 'krea2', comfy: true, textEncoderArray: true, fields: [file('diffusion_model', '扩散模型权重', 'Diffusion model'), file('vae', 'VAE 权重', 'VAE weights'), file('text_encoder', '文本编码器权重', 'Text encoder weights')] },
}

export const MODEL_ARCHITECTURES = Object.keys(MODEL_PROFILES)

export function getModelProfile(name) {
  return MODEL_PROFILES[name] || MODEL_PROFILES.Anima
}
