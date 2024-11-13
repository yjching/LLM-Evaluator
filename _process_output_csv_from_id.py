import pandas as pd
import os

input_df = pd.read_csv(f"{os.getcwd()}/LLM-Evaluator/data/rag_chatbot_guardrails_output.csv", encoding='utf-8')
input_df.columns

df = input_df[["Final_Response", "_score_", "answer_llm", "complete_prompt", "sources", "content"]]
df

df['sources']

## Join with the actual responses
actual_df = pd.read_csv(f"{os.getcwd()}/LLM-Evaluator/data/test_sample_prompts_chatbot_responses.csv")
actual_df['answer']
actual_df.columns
merged_df = pd.merge(df, actual_df, on='content')
merged_df.columns

merged_df.to_csv(f"{os.getcwd()}/LLM-Evaluator/data/rag_chatbot_guardrails_evaluationset.csv", index=False)