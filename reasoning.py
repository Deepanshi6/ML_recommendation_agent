from dotenv import load_dotenv
import os
import json
import pandas as pd
from groq import Groq

load_dotenv()

llm_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=llm_api_key)

def data_manage(df, target):

    print("Reasoning if some columns need to be dropped such as ID , name.")

    data_info = {
        "Columns": df.columns.tolist(),
        "target": target
    }

    prompt = f"""
You are a data management agent.

Your job is to identify columns that are unlikely to be useful
for further Machine Learning processing, such as:
- ID columns
- unique identifiers
- names
- transaction/reference numbers
- columns with almost every value unique

Do NOT remove the target column.

Data:
{json.dumps(data_info, default=str, indent=2)}

Return ONLY valid JSON.

Format:
{{
    "columns": [
        {{
            "column": "column_name",
            "reason": "reason for removal"
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a strict dataframe data management agent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def columns_check(df):

    print("Reasoning if some column type needs to be changed ......")
    columns = [
        {
            "column": col,
            "pandas_dtype": str(df[col].dtype),
            "sample_values": df[col].dropna().head(10).tolist()
        }
        for col in df.columns
    ]

    prompt = f"""
You are a dataframe datatype validation agent.

Identify ONLY columns whose pandas dtype should be changed
for proper ML/data processing.

Possible types:
string, integer, float, boolean, date, datetime, categorical, monetary.

Use column name, pandas dtype, and sample values to determine
the semantic type.

Return ONLY columns that need a dtype conversion.

Return valid JSON in this format:
example of format:
{{
    "columns": [
        {{
            "column": "date_of_admission",
            "current_dtype": "object",
            "detected_type": "date",
            "suggested_dtype": "datetime64[ns]"
        }}
    ]
}}

If no columns need changes, return:
{{"columns": []}}

Data:
{json.dumps(columns, default=str, indent=2)}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def choose_boosting_algorithm(df, target,target_type,total_columns,numerical_columns,categorical_columns,null_columns):

    data_info={"target":target,
        "target_type":target_type,
               "total_columns":total_columns,
               "total_rows":len(df),
               "numerical_columns":numerical_columns,
               "categorical_columns":categorical_columns,
               "null_columns":null_columns}

    prompt = f'''
    You are strictly an Machine Learning algorithm suggestion agent .
    Identify the best 2 best Machine Learning algorithms with their respective reason on basis of data_information.
    Also you can add some extra remark for the user to consider for further machine learning model training.

    data_information: 
    {json.dumps(data_info,default=str,indent=2)}

    Return:
    - Best_algorithm
    - Reason_of_best_algorithm
    - 2nd_best_algorithm
    - Reason_of_2nd_best_algorithm
    - Remark_further_ml

    Format:
    {{
        "Best algorithm":Best_algorithm,
        "Reason_of_best_algorithm":Reason_of_best_algorithm,
        "2nd_best_algorithm":2nd_best_algorithm,
        "Reason_of_2nd_best_algorithm":Reason_of_2nd_best_algorithm,
        "Remark_for_further_Ml_processing":Remark_further_ml
    }}
            '''
    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict Machine Learning Algorithm suggestion agent."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    
    return json.loads(response.choices[0].message.content)

REASONING={
    "data_manage":data_manage,
    "columns_check":columns_check,
    "choose_boosting_algorithm":choose_boosting_algorithm
}
