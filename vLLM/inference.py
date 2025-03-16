from vllm import LLM, SamplingParams

prompts = [
    "我最近因为考试，觉得心情很不好，自己没有及格，请安慰我一下可以吗？"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/mnt/workspace/LLaMA-Factory/emo-glm4/glm4-9b-export", 
          trust_remote_code=True, # 允许执行远程代码
          max_model_len = 2048, # 控制模型支持的最大序列长度,可以通过减少该值来降低对GPU内存的需求
          gpu_memory_utilization = 0.9) # 控制 GPU 内存中有多少比例用于存储 KV 缓存

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print("Prompt: ", prompt)
    print("Generated text: ", generated_text)
