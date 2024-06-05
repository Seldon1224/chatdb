import argparse
import sqlparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn.parallel import DataParallel

def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model

# 加载模型和分词器
model_name = "defog/llama-3-sqlcoder-8b"
tokenizer, model = get_tokenizer_model(model_name)


def format_sql(sql_query):
    formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
    return formatted_sql

def generate_prompt(user_question, prompt_file, metadata_file):
    with open(metadata_file, "r") as f:
        create_table_statements = f.read()
    
    with open(prompt_file, "r") as f:
        prompt = f.read()

    prompt = prompt.format(
        user_question=user_question, create_table_statements=create_table_statements
    )
    return prompt


def run_inference(question, prompt_file="prompt_0603.md", metadata_file="metadata_xyq.sql"):
    prompt = generate_prompt(question, prompt_file, metadata_file)
    # make sure the model stops generating at triple ticks
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline(
        "text-generation",
        model=model,
        return_full_text=False,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=True,
        num_beams=10, # do beam search with 5 beams for high quality results
    )

    time1 = time.time()
    with torch.cuda.amp.autocast():
        generated_output = pipe(
            prompt,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id
        )

    # 打印生成的结果以进行调试
    #print("Generated Output:\n", generated_output)
    generated_text = generated_output[0]["generated_text"]
    # 去除prompt部分，保留生成的SQL查询
    generated_query = generated_text.replace(prompt, "").split(";")[0].strip() + ";"
    elapsed_time = time.time() - time1
    print(f"Elapsed time: {elapsed_time} seconds")
    return format_sql(generated_query)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference on a question")
    parser.add_argument("-q", "--question", type=str, help="Question to run inference on")
    parser.add_argument("-m", "--meta_file", type=str, help="meta_file to run inference on")
    args = parser.parse_args()
    question = args.question
    print("Loading a model and generating a SQL query for answering your question...")

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        print("Generating SQL query for your question...")
        if args.meta_file is None or args.meta_file == '':
            sql_query = run_inference(question)
        else:
            print(f"Using meta_file: {args.meta_file}...")
            sql_query = run_inference(question, metadata_file=args.meta_file)
        print(f"Question: {question}\nGenerated SQL Query: \n{sql_query}\n")
    


    
