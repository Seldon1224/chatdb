
### 代码详解

#### 1. 导入库和加载模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "defog/llama-3-sqlcoder-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

- `torch`: PyTorch库，用于深度学习模型的加载和推理。
- `AutoModelForCausalLM`: 从Transformers库中导入的类，用于加载因果语言模型。
- `AutoTokenizer`: 从Transformers库中导入的类，用于加载分词器。
- `model_name`: 模型的名称，这里使用的是"defog/llama-3-sqlcoder-8b"。

#### 2. 添加填充标记

```python
# 添加填充标记
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
```

- `tokenizer.pad_token`: 检查分词器是否已经有一个填充标记。
- `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`: 如果没有填充标记，则添加一个新的填充标记`[PAD]`。

#### 3. 调整模型的词嵌入

```python
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
```

- `AutoModelForCausalLM.from_pretrained(model_name)`: 加载预训练的因果语言模型。
- `model.resize_token_embeddings(len(tokenizer))`: 调整模型的词嵌入矩阵大小，以适应新的填充标记。

#### 4. 定义生成SQL查询的函数

```python
def generate_sql_query(user_question, create_table_statements):
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate a SQL query to answer this question: `{user_question}`\n\nDDL statements:\n{create_table_statements}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe following SQL query best answers the question `{user_question}`:\n```sql\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    sql_query = generated_text.split("```sql\n")[1].split("\n```")[0]
    return sql_query
```

- `prompt`: 构建提示字符串，用于指导模型生成SQL查询。
- `tokenizer(prompt, return_tensors="pt", padding=True)`: 将提示字符串转换为张量，并进行填充。
  - `return_tensors="pt"`: 返回PyTorch张量。
  - `padding=True`: 启用填充。
- `model.generate`: 使用模型生成文本。
  - `inputs["input_ids"]`: 输入张量的ID。
  - `attention_mask=inputs["attention_mask"]`: 注意力掩码，用于指示填充部分。
  - `max_length=512`: 生成文本的最大长度。
  - `pad_token_id=tokenizer.pad_token_id`: 填充标记的ID。
- `tokenizer.decode(output[0], skip_special_tokens=True)`: 将生成的张量转换回文本。
  - `skip_special_tokens=True`: 跳过特殊标记。
- `split("```sql\n")[1].split("\n```")[0]`: 从生成的文本中提取SQL查询语句。

#### 5. 示例使用

```python
# 示例使用
user_question = "获取所有用户的名字和电子邮件"
create_table_statements = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);
"""

sql_query = generate_sql_query(user_question, create_table_statements)
print("生成的SQL查询语句:")
print(sql_query)
```

- `user_question`: 用户的问题。
- `create_table_statements`: DDL语句，描述数据库表的结构。
- `generate_sql_query(user_question, create_table_statements)`: 调用函数生成SQL查询。
- `print(sql_query)`: 打印生成的SQL查询语句。

### 参数调整

1. **`max_length`**:
   - 默认值是512，表示生成的文本的最大长度。你可以根据需要调整这个值，例如，如果你希望生成更短或更长的查询语句，可以修改这个值。

   ```python
   max_length=256  # 生成更短的文本
   ```

2. **`temperature`**:
   - 控制生成的随机性。较高的温度会生成更加多样化的文本，较低的温度会生成更加确定性的文本。默认情况下，我们没有设置这个参数，因为我们不进行采样。

   ```python
   temperature=0.7  # 增加生成文本的多样性
   ```

3. **`do_sample`**:
   - 是否进行采样。默认是`False`，如果你希望生成更加多样化的文本，可以将其设置为`True`。

   ```python
   do_sample=True  # 启用采样
   ```

4. **`top_p`**:
   - 控制生成时的累积概率阈值，用于核采样。仅在`do_sample=True`时有效。

   ```python
   top_p=0.9  # 仅生成累积概率在0.9以内的文本
   ```

通过调整这些参数，你可以控制生成文本的长度、随机性和多样性，以达到不同的预期效果。如果你有其他特定需求或问题，请随时告诉我。


